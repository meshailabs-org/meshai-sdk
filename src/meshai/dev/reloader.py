"""
Hot Reloader for MeshAI Development Server

Handles module reloading, dependency management, and state preservation.
"""

import asyncio
import logging
import sys
import importlib
import inspect
import gc
import traceback
from typing import Dict, Set, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import threading
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Information about a loaded module"""
    name: str
    file_path: str
    load_time: datetime
    dependencies: Set[str]
    dependents: Set[str]
    instances: List[weakref.ref]


class HotReloader:
    """
    Intelligent hot reloader for Python modules
    
    Features:
    - Safe module reloading with dependency tracking
    - Instance state preservation
    - Error recovery and rollback
    - Circular dependency handling
    - Memory leak prevention
    """
    
    def __init__(
        self,
        watch_paths: List[str],
        reload_callback: Optional[Callable] = None,
        preserve_state: bool = True,
        safe_mode: bool = True
    ):
        """
        Initialize hot reloader
        
        Args:
            watch_paths: Paths containing modules to reload
            reload_callback: Callback to execute after reload
            preserve_state: Try to preserve instance state across reloads
            safe_mode: Enable safe reloading with rollback on errors
        """
        self.watch_paths = [Path(p).resolve() for p in watch_paths]
        self.reload_callback = reload_callback
        self.preserve_state = preserve_state
        self.safe_mode = safe_mode
        
        # Module tracking
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_snapshots: Dict[str, Any] = {}
        self.reload_queue: Set[str] = set()
        self.reloading = False
        
        # State preservation
        self.preserved_instances: Dict[str, Dict[str, Any]] = {}
        self.instance_registry: Dict[str, List[weakref.ref]] = {}
        
        # Statistics
        self.stats = {
            "reloads_attempted": 0,
            "reloads_successful": 0,
            "reloads_failed": 0,
            "modules_tracked": 0,
            "last_reload": None
        }
        
        # Initialize tracking for existing modules
        self._initialize_module_tracking()
    
    def _initialize_module_tracking(self):
        """Initialize tracking for already loaded modules"""
        
        for name, module in sys.modules.items():
            if self._should_track_module(name, module):
                self._add_module_tracking(name, module)
    
    def _should_track_module(self, name: str, module: Any) -> bool:
        """Check if a module should be tracked for reloading"""
        
        if not hasattr(module, '__file__') or not module.__file__:
            return False
        
        module_path = Path(module.__file__).resolve()
        
        # Check if module is in watch paths
        for watch_path in self.watch_paths:
            try:
                module_path.relative_to(watch_path)
                return True
            except ValueError:
                continue
        
        return False
    
    def _add_module_tracking(self, name: str, module: Any):
        """Add a module to tracking"""
        
        if not hasattr(module, '__file__'):
            return
        
        file_path = str(Path(module.__file__).resolve())
        
        self.modules[name] = ModuleInfo(
            name=name,
            file_path=file_path,
            load_time=datetime.now(),
            dependencies=self._get_module_dependencies(module),
            dependents=set(),
            instances=[]
        )
        
        self.stats["modules_tracked"] += 1
        logger.debug(f"Tracking module: {name}")
    
    def _get_module_dependencies(self, module: Any) -> Set[str]:
        """Get dependencies of a module"""
        
        dependencies = set()
        
        if not hasattr(module, '__file__'):
            return dependencies
        
        try:
            # Get imports from module globals
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if inspect.ismodule(attr):
                    dep_name = getattr(attr, '__name__', None)
                    if dep_name and self._should_track_module(dep_name, attr):
                        dependencies.add(dep_name)
                
                elif inspect.isclass(attr) or inspect.isfunction(attr):
                    # Check if defined in a different tracked module
                    if hasattr(attr, '__module__'):
                        dep_name = attr.__module__
                        if dep_name != module.__name__ and dep_name in sys.modules:
                            dep_module = sys.modules[dep_name]
                            if self._should_track_module(dep_name, dep_module):
                                dependencies.add(dep_name)
        
        except Exception as e:
            logger.debug(f"Error getting dependencies for {module}: {e}")
        
        return dependencies
    
    def _update_dependents(self):
        """Update dependent relationships between modules"""
        
        for name, module_info in self.modules.items():
            module_info.dependents.clear()
        
        for name, module_info in self.modules.items():
            for dep_name in module_info.dependencies:
                if dep_name in self.modules:
                    self.modules[dep_name].dependents.add(name)
    
    def register_instance(self, instance: Any, module_name: str):
        """Register an instance for state preservation"""
        
        if not self.preserve_state:
            return
        
        instance_id = f"{module_name}.{instance.__class__.__name__}.{id(instance)}"
        
        if module_name not in self.instance_registry:
            self.instance_registry[module_name] = []
        
        # Store weak reference to avoid memory leaks
        weak_ref = weakref.ref(instance, self._cleanup_instance_ref)
        self.instance_registry[module_name].append(weak_ref)
        
        # Store initial state
        if instance_id not in self.preserved_instances:
            self.preserved_instances[instance_id] = {}
            
        logger.debug(f"Registered instance: {instance_id}")
    
    def _cleanup_instance_ref(self, weak_ref: weakref.ref):
        """Clean up dead instance references"""
        
        for module_name, refs in self.instance_registry.items():
            if weak_ref in refs:
                refs.remove(weak_ref)
                break
    
    async def reload(self, module_names: Optional[List[str]] = None):
        """
        Reload modules with dependency resolution
        
        Args:
            module_names: Specific modules to reload (None for all changed)
        """
        
        if self.reloading:
            logger.warning("Reload already in progress")
            return
        
        self.reloading = True
        self.stats["reloads_attempted"] += 1
        
        try:
            logger.info("🔄 Starting hot reload...")
            
            # Determine modules to reload
            if module_names is None:
                modules_to_reload = list(self.reload_queue)
                self.reload_queue.clear()
            else:
                modules_to_reload = module_names
            
            if not modules_to_reload:
                logger.debug("No modules to reload")
                return
            
            # Add dependents to reload queue
            all_modules_to_reload = set(modules_to_reload)
            for module_name in modules_to_reload:
                if module_name in self.modules:
                    all_modules_to_reload.update(self.modules[module_name].dependents)
            
            # Sort by dependency order
            reload_order = self._get_reload_order(list(all_modules_to_reload))
            
            logger.info(f"Reloading {len(reload_order)} modules: {reload_order}")
            
            # Take snapshots for rollback
            if self.safe_mode:
                await self._take_snapshots(reload_order)
            
            # Preserve instance state
            if self.preserve_state:
                await self._preserve_instance_state(reload_order)
            
            # Perform reload
            success = await self._reload_modules(reload_order)
            
            if success:
                # Restore instance state
                if self.preserve_state:
                    await self._restore_instance_state(reload_order)
                
                # Execute callback
                if self.reload_callback:
                    if asyncio.iscoroutinefunction(self.reload_callback):
                        await self.reload_callback()
                    else:
                        self.reload_callback()
                
                self.stats["reloads_successful"] += 1
                self.stats["last_reload"] = datetime.now()
                logger.info("✅ Hot reload completed successfully")
                
            else:
                self.stats["reloads_failed"] += 1
                logger.error("❌ Hot reload failed")
                
                # Rollback if in safe mode
                if self.safe_mode:
                    await self._rollback_modules(reload_order)
        
        except Exception as e:
            logger.error(f"Hot reload error: {e}")
            logger.debug(traceback.format_exc())
            self.stats["reloads_failed"] += 1
            
            # Rollback on error
            if self.safe_mode:
                try:
                    await self._rollback_modules(reload_order)
                except:
                    logger.error("Rollback also failed")
        
        finally:
            self.reloading = False
    
    def _get_reload_order(self, module_names: List[str]) -> List[str]:
        """Get modules in dependency order for safe reloading"""
        
        # Topological sort
        in_degree = {name: 0 for name in module_names}
        graph = {name: set() for name in module_names}
        
        # Build dependency graph
        for name in module_names:
            if name in self.modules:
                for dep in self.modules[name].dependencies:
                    if dep in module_names:
                        graph[dep].add(name)
                        in_degree[name] += 1
        
        # Kahn's algorithm
        queue = [name for name in module_names if in_degree[name] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(module_names):
            logger.warning("Circular dependencies detected, using original order")
            return module_names
        
        return result
    
    async def _take_snapshots(self, module_names: List[str]):
        """Take snapshots of modules for rollback"""
        
        for name in module_names:
            if name in sys.modules:
                module = sys.modules[name]
                
                # Create shallow copy of module dict
                self.module_snapshots[name] = {
                    'dict': dict(module.__dict__),
                    'file': getattr(module, '__file__', None)
                }
    
    async def _preserve_instance_state(self, module_names: List[str]):
        """Preserve state of instances from modules being reloaded"""
        
        for module_name in module_names:
            if module_name not in self.instance_registry:
                continue
            
            # Get live instances
            instances = []
            for weak_ref in self.instance_registry[module_name]:
                instance = weak_ref()
                if instance is not None:
                    instances.append(instance)
            
            # Preserve state
            for instance in instances:
                instance_id = f"{module_name}.{instance.__class__.__name__}.{id(instance)}"
                
                try:
                    # Store serializable attributes
                    state = {}
                    for attr_name in dir(instance):
                        if attr_name.startswith('_'):
                            continue
                        
                        try:
                            attr_value = getattr(instance, attr_name)
                            if self._is_serializable(attr_value):
                                state[attr_name] = attr_value
                        except:
                            continue
                    
                    self.preserved_instances[instance_id] = state
                    logger.debug(f"Preserved state for {instance_id}")
                    
                except Exception as e:
                    logger.debug(f"Could not preserve state for {instance_id}: {e}")
    
    def _is_serializable(self, obj: Any) -> bool:
        """Check if an object is serializable for state preservation"""
        
        import json
        
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    async def _reload_modules(self, module_names: List[str]) -> bool:
        """Reload the specified modules"""
        
        try:
            for name in module_names:
                if name not in sys.modules:
                    logger.debug(f"Module {name} not in sys.modules, skipping")
                    continue
                
                logger.debug(f"Reloading module: {name}")
                
                try:
                    # Reload the module
                    importlib.reload(sys.modules[name])
                    
                    # Update tracking info
                    if name in self.modules:
                        module = sys.modules[name]
                        self.modules[name].dependencies = self._get_module_dependencies(module)
                        self.modules[name].load_time = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Failed to reload module {name}: {e}")
                    return False
            
            # Update dependent relationships
            self._update_dependents()
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Module reload failed: {e}")
            return False
    
    async def _restore_instance_state(self, module_names: List[str]):
        """Restore preserved instance state after reload"""
        
        for module_name in module_names:
            if module_name not in self.instance_registry:
                continue
            
            # Get current instances (may be new after reload)
            instances = []
            for weak_ref in self.instance_registry[module_name]:
                instance = weak_ref()
                if instance is not None:
                    instances.append(instance)
            
            # Restore state
            for instance in instances:
                instance_id = f"{module_name}.{instance.__class__.__name__}.{id(instance)}"
                
                if instance_id in self.preserved_instances:
                    state = self.preserved_instances[instance_id]
                    
                    try:
                        for attr_name, attr_value in state.items():
                            if hasattr(instance, attr_name):
                                setattr(instance, attr_name, attr_value)
                        
                        logger.debug(f"Restored state for {instance_id}")
                        
                    except Exception as e:
                        logger.debug(f"Could not restore state for {instance_id}: {e}")
    
    async def _rollback_modules(self, module_names: List[str]):
        """Rollback modules to their previous state"""
        
        logger.warning("🔄 Rolling back modules...")
        
        for name in reversed(module_names):  # Rollback in reverse order
            if name in self.module_snapshots:
                snapshot = self.module_snapshots[name]
                
                try:
                    if name in sys.modules:
                        # Restore module dict
                        module = sys.modules[name]
                        module.__dict__.clear()
                        module.__dict__.update(snapshot['dict'])
                        
                        logger.debug(f"Rolled back module: {name}")
                    
                except Exception as e:
                    logger.error(f"Rollback failed for {name}: {e}")
        
        logger.info("↩️ Rollback completed")
    
    def schedule_reload(self, module_name: str):
        """Schedule a module for reloading"""
        
        self.reload_queue.add(module_name)
        logger.debug(f"Scheduled reload for: {module_name}")
    
    def get_module_from_file(self, file_path: str) -> Optional[str]:
        """Get module name from file path"""
        
        file_path = str(Path(file_path).resolve())
        
        for name, module_info in self.modules.items():
            if module_info.file_path == file_path:
                return name
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reloader statistics"""
        
        return {
            **self.stats,
            "modules_tracked": len(self.modules),
            "reload_queue_size": len(self.reload_queue),
            "preserved_instances": len(self.preserved_instances),
            "is_reloading": self.reloading
        }


class StatePreserver:
    """
    Advanced state preservation for hot reloading
    
    Handles complex state preservation scenarios including:
    - Async contexts and coroutines
    - Database connections
    - File handles
    - Complex object hierarchies
    """
    
    def __init__(self):
        self.preservers: Dict[type, Callable] = {}
        self.restorers: Dict[type, Callable] = {}
        
        # Register default preservers
        self._register_default_preservers()
    
    def _register_default_preservers(self):
        """Register default state preservers for common types"""
        
        # Basic types are automatically handled
        
        # Custom preservers for complex types would go here
        pass
    
    def register_preserver(self, type_class: type, preserver: Callable, restorer: Callable):
        """Register custom state preserver for a type"""
        
        self.preservers[type_class] = preserver
        self.restorers[type_class] = restorer
    
    def preserve(self, obj: Any) -> Dict[str, Any]:
        """Preserve state of an object"""
        
        obj_type = type(obj)
        
        if obj_type in self.preservers:
            return self.preservers[obj_type](obj)
        
        # Default preservation
        return self._default_preserve(obj)
    
    def restore(self, obj: Any, state: Dict[str, Any]):
        """Restore state to an object"""
        
        obj_type = type(obj)
        
        if obj_type in self.restorers:
            self.restorers[obj_type](obj, state)
        else:
            self._default_restore(obj, state)
    
    def _default_preserve(self, obj: Any) -> Dict[str, Any]:
        """Default state preservation"""
        
        state = {}
        
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                if not callable(attr_value):
                    state[attr_name] = attr_value
            except:
                continue
        
        return state
    
    def _default_restore(self, obj: Any, state: Dict[str, Any]):
        """Default state restoration"""
        
        for attr_name, attr_value in state.items():
            try:
                if hasattr(obj, attr_name):
                    setattr(obj, attr_name, attr_value)
            except:
                continue


# Testing and example usage
if __name__ == "__main__":
    
    async def test_reloader():
        """Test the hot reloader"""
        
        def reload_callback():
            print("Reload completed!")
        
        reloader = HotReloader(
            watch_paths=["."],
            reload_callback=reload_callback,
            preserve_state=True,
            safe_mode=True
        )
        
        # Schedule a reload
        reloader.schedule_reload("test_module")
        
        # Perform reload
        await reloader.reload()
        
        # Print stats
        stats = reloader.get_stats()
        print(f"Reloader stats: {stats}")
    
    # Run test
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_reloader())