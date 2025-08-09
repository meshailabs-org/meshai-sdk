"""
File Watcher for MeshAI Development Server

Monitors file system changes and triggers hot-reload.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List, Callable, Optional, Set, Dict, Any
import threading
from dataclasses import dataclass
from enum import Enum

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of file system changes"""
    CREATED = "created"
    MODIFIED = "modified" 
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChange:
    """Represents a file system change"""
    path: str
    change_type: ChangeType
    timestamp: float
    is_directory: bool = False


class FileWatcher:
    """
    File system watcher with debouncing and filtering
    
    Features:
    - Cross-platform file watching
    - Debouncing to prevent duplicate events
    - File type filtering
    - Recursive directory monitoring
    - Async callback support
    """
    
    def __init__(
        self,
        paths: List[str],
        callback: Callable[[str, str], None],
        debounce_ms: int = 500,
        file_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ):
        """
        Initialize file watcher
        
        Args:
            paths: Paths to watch
            callback: Callback function for file changes
            debounce_ms: Debounce time in milliseconds
            file_patterns: File patterns to watch (e.g., ['*.py', '*.yaml'])
            ignore_patterns: Patterns to ignore (e.g., ['*.pyc', '__pycache__'])
            recursive: Watch subdirectories recursively
        """
        self.paths = [Path(p).resolve() for p in paths]
        self.callback = callback
        self.debounce_ms = debounce_ms
        self.recursive = recursive
        
        # File filtering
        self.file_patterns = file_patterns or ['*.py', '*.yaml', '*.yml', '*.json', '*.toml']
        self.ignore_patterns = ignore_patterns or [
            '*.pyc', '*.pyo', '*.pyd', '__pycache__', '.git', '.venv', 
            'node_modules', '.pytest_cache', '*.log', '.DS_Store'
        ]
        
        # State tracking
        self.observer = None
        self.is_running = False
        self._change_queue: Dict[str, FileChange] = {}
        self._debounce_task = None
        self._loop = None
        
        # Performance tracking
        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "events_debounced": 0,
            "events_filtered": 0
        }
    
    def start(self):
        """Start watching for file changes"""
        
        if self.is_running:
            return
        
        logger.info(f"👁️ Starting file watcher for {len(self.paths)} paths")
        
        if not HAS_WATCHDOG:
            logger.warning("watchdog not available, falling back to polling")
            self._start_polling()
        else:
            self._start_watchdog()
        
        self.is_running = True
    
    def stop(self):
        """Stop watching for file changes"""
        
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping file watcher")
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        self.is_running = False
    
    def _start_watchdog(self):
        """Start watchdog-based file watching"""
        
        class MeshAIEventHandler(FileSystemEventHandler):
            def __init__(self, watcher: FileWatcher):
                self.watcher = watcher
                super().__init__()
            
            def on_any_event(self, event: FileSystemEvent):
                if event.is_directory:
                    return
                    
                self.watcher._handle_event(
                    event.src_path,
                    self._map_event_type(event.event_type),
                    event.is_directory
                )
            
            def _map_event_type(self, event_type: str) -> ChangeType:
                mapping = {
                    'created': ChangeType.CREATED,
                    'modified': ChangeType.MODIFIED,
                    'deleted': ChangeType.DELETED,
                    'moved': ChangeType.MOVED
                }
                return mapping.get(event_type, ChangeType.MODIFIED)
        
        self.observer = Observer()
        handler = MeshAIEventHandler(self)
        
        for path in self.paths:
            if path.exists():
                self.observer.schedule(handler, str(path), recursive=self.recursive)
                logger.debug(f"Watching: {path}")
        
        self.observer.start()
    
    def _start_polling(self):
        """Start polling-based file watching (fallback)"""
        
        def poll_files():
            file_states = {}
            
            while self.is_running:
                try:
                    current_states = {}
                    
                    for watch_path in self.paths:
                        if not watch_path.exists():
                            continue
                            
                        for file_path in self._get_files_to_watch(watch_path):
                            try:
                                stat = file_path.stat()
                                current_states[str(file_path)] = {
                                    'mtime': stat.st_mtime,
                                    'size': stat.st_size
                                }
                            except (OSError, FileNotFoundError):
                                # File was deleted or is inaccessible
                                if str(file_path) in file_states:
                                    self._handle_event(
                                        str(file_path), 
                                        ChangeType.DELETED,
                                        False
                                    )
                    
                    # Check for changes
                    for file_path, current_state in current_states.items():
                        old_state = file_states.get(file_path)
                        
                        if old_state is None:
                            # New file
                            self._handle_event(file_path, ChangeType.CREATED, False)
                        elif (current_state['mtime'] != old_state['mtime'] or 
                              current_state['size'] != old_state['size']):
                            # Modified file
                            self._handle_event(file_path, ChangeType.MODIFIED, False)
                    
                    file_states = current_states
                    time.sleep(1)  # Poll every second
                    
                except Exception as e:
                    logger.error(f"Error in file polling: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=poll_files, daemon=True)
        thread.start()
    
    def _get_files_to_watch(self, root_path: Path) -> List[Path]:
        """Get list of files to watch in a directory"""
        
        files = []
        
        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in root_path.glob(pattern):
            if file_path.is_file() and self._should_watch_file(file_path):
                files.append(file_path)
        
        return files
    
    def _handle_event(self, file_path: str, change_type: ChangeType, is_directory: bool):
        """Handle a file system event"""
        
        self.stats["events_received"] += 1
        
        # Filter unwanted files
        if not self._should_watch_file(Path(file_path)):
            self.stats["events_filtered"] += 1
            return
        
        # Create change record
        change = FileChange(
            path=file_path,
            change_type=change_type,
            timestamp=time.time(),
            is_directory=is_directory
        )
        
        # Add to debounce queue
        self._change_queue[file_path] = change
        
        # Start or restart debounce timer
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            self.stats["events_debounced"] += 1
        
        # Schedule debounced processing
        if not self._loop:
            # Try to get current event loop
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create one
                logger.debug("Creating new event loop for file watcher")
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        
        self._debounce_task = self._loop.call_later(
            self.debounce_ms / 1000.0,
            self._process_changes
        )
    
    def _process_changes(self):
        """Process queued file changes after debounce period"""
        
        if not self._change_queue:
            return
        
        changes = list(self._change_queue.values())
        self._change_queue.clear()
        
        logger.debug(f"Processing {len(changes)} file changes")
        
        # Group changes by type for efficient processing
        for change in changes:
            self.stats["events_processed"] += 1
            
            try:
                # Call the callback
                if asyncio.iscoroutinefunction(self.callback):
                    # Async callback
                    if self._loop and self._loop.is_running():
                        self._loop.create_task(
                            self.callback(change.path, change.change_type.value)
                        )
                    else:
                        # Run in new thread
                        threading.Thread(
                            target=lambda: asyncio.run(
                                self.callback(change.path, change.change_type.value)
                            ),
                            daemon=True
                        ).start()
                else:
                    # Sync callback
                    self.callback(change.path, change.change_type.value)
                    
            except Exception as e:
                logger.error(f"Error in file change callback: {e}")
    
    def _should_watch_file(self, file_path: Path) -> bool:
        """Check if a file should be watched"""
        
        file_name = file_path.name
        file_str = str(file_path)
        
        # Check ignore patterns first
        for pattern in self.ignore_patterns:
            if self._match_pattern(file_str, pattern) or self._match_pattern(file_name, pattern):
                return False
        
        # Check if it matches any include patterns
        if not self.file_patterns:
            return True
            
        for pattern in self.file_patterns:
            if self._match_pattern(file_name, pattern):
                return True
        
        return False
    
    def _match_pattern(self, text: str, pattern: str) -> bool:
        """Simple pattern matching (supports * and ? wildcards)"""
        
        import fnmatch
        return fnmatch.fnmatch(text, pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics"""
        
        return {
            **self.stats,
            "is_running": self.is_running,
            "paths_watched": len(self.paths),
            "queue_size": len(self._change_queue),
            "backend": "watchdog" if HAS_WATCHDOG else "polling"
        }


class SmartFileWatcher:
    """
    Enhanced file watcher with intelligent change detection
    
    Features:
    - Content-aware change detection
    - Dependency tracking
    - Performance optimization
    - Change impact analysis
    """
    
    def __init__(self, base_watcher: FileWatcher):
        self.base_watcher = base_watcher
        self.file_hashes: Dict[str, str] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Wrap the original callback
        self.original_callback = base_watcher.callback
        base_watcher.callback = self._smart_callback
    
    async def _smart_callback(self, file_path: str, event_type: str):
        """Enhanced callback with content checking and dependency analysis"""
        
        # Check if the file content actually changed
        if event_type == "modified":
            if not await self._content_changed(file_path):
                logger.debug(f"No content change detected for {file_path}")
                return
        
        # Analyze change impact
        affected_files = self._get_affected_files(file_path)
        
        # Call original callback
        await self.original_callback(file_path, event_type)
        
        # Notify about dependent files
        for affected_file in affected_files:
            await self.original_callback(affected_file, "dependency_changed")
    
    async def _content_changed(self, file_path: str) -> bool:
        """Check if file content actually changed using hash comparison"""
        
        try:
            import hashlib
            
            with open(file_path, 'rb') as f:
                content = f.read()
                new_hash = hashlib.md5(content).hexdigest()
            
            old_hash = self.file_hashes.get(file_path)
            self.file_hashes[file_path] = new_hash
            
            return old_hash is None or old_hash != new_hash
            
        except Exception as e:
            logger.debug(f"Error checking content change for {file_path}: {e}")
            return True  # Assume changed if we can't check
    
    def _get_affected_files(self, file_path: str) -> Set[str]:
        """Get files that might be affected by changes to the given file"""
        
        # Simple implementation - could be enhanced with AST analysis
        affected = set()
        
        for dependent, dependencies in self.dependency_graph.items():
            if file_path in dependencies:
                affected.add(dependent)
        
        return affected
    
    def add_dependency(self, file_path: str, depends_on: str):
        """Add a dependency relationship between files"""
        
        if file_path not in self.dependency_graph:
            self.dependency_graph[file_path] = set()
        
        self.dependency_graph[file_path].add(depends_on)


# Usage example and testing
if __name__ == "__main__":
    import sys
    
    async def test_callback(file_path: str, event_type: str):
        print(f"File changed: {file_path} ({event_type})")
    
    def run_test():
        watcher = FileWatcher(
            paths=["."],
            callback=test_callback,
            debounce_ms=1000
        )
        
        try:
            watcher.start()
            print("File watcher started. Press Ctrl+C to stop.")
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping file watcher...")
            watcher.stop()
            
            # Print stats
            stats = watcher.get_stats()
            print(f"Statistics: {stats}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test()