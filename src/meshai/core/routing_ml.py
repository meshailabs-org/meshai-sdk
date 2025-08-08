"""
Machine Learning-Enhanced Routing Decision Engine

This module provides ML-powered routing decisions that learn from
historical performance data to optimize agent selection.
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import pickle
import os

import structlog
from .routing_engine import RoutingContext, RoutingDecision, AgentEndpoint, RoutingStrategy

logger = structlog.get_logger(__name__)


@dataclass
class FeatureVector:
    """Feature vector for ML routing decisions"""
    # Request features
    request_priority: float = 0.0
    request_timeout: float = 0.0
    request_size: float = 0.0  # Estimated request complexity
    retry_count: float = 0.0
    
    # Agent features
    agent_avg_response_time: float = 0.0
    agent_success_rate: float = 0.0
    agent_current_load: float = 0.0
    agent_capacity_utilization: float = 0.0
    agent_recent_errors: float = 0.0
    
    # Context features
    hour_of_day: float = 0.0
    day_of_week: float = 0.0
    recent_request_volume: float = 0.0
    
    # Historical features
    agent_performance_trend: float = 0.0
    success_rate_last_hour: float = 0.0
    avg_response_time_last_hour: float = 0.0


@dataclass
class TrainingExample:
    """Training example for ML model"""
    features: FeatureVector
    agent_id: str
    actual_response_time: float
    success: bool
    timestamp: datetime
    reward: float = 0.0  # Calculated reward for this decision


@dataclass
class ModelPrediction:
    """ML model prediction for agent selection"""
    agent_id: str
    predicted_response_time: float
    predicted_success_probability: float
    confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)


class SimpleLinearModel:
    """Simple linear regression model for routing decisions"""
    
    def __init__(self, feature_dim: int = 13):
        self.feature_dim = feature_dim
        self.weights = np.random.normal(0, 0.1, feature_dim)
        self.bias = 0.0
        self.learning_rate = 0.01
        self.l2_reg = 0.001
        self.trained = False
    
    def predict(self, features: np.ndarray) -> float:
        """Predict response time"""
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {features.shape[0]}")
        
        return np.dot(self.weights, features) + self.bias
    
    def train_batch(self, features_batch: np.ndarray, targets_batch: np.ndarray):
        """Train model on a batch of examples"""
        if len(features_batch) == 0:
            return
        
        # Predictions
        predictions = np.dot(features_batch, self.weights) + self.bias
        
        # Compute gradients
        error = predictions - targets_batch
        n_samples = len(features_batch)
        
        # Update weights with L2 regularization
        weight_grad = np.dot(features_batch.T, error) / n_samples + self.l2_reg * self.weights
        bias_grad = np.mean(error)
        
        self.weights -= self.learning_rate * weight_grad
        self.bias -= self.learning_rate * bias_grad
        
        self.trained = True
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (absolute weights)"""
        return np.abs(self.weights)


class EnsembleModel:
    """Ensemble of simple models for better predictions"""
    
    def __init__(self, n_models: int = 5, feature_dim: int = 13):
        self.models = [SimpleLinearModel(feature_dim) for _ in range(n_models)]
        self.feature_dim = feature_dim
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict with confidence (mean, std)"""
        predictions = [model.predict(features) for model in self.models if model.trained]
        
        if not predictions:
            return 1.0, 1.0  # Default prediction with high uncertainty
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions) if len(predictions) > 1 else 0.5
        
        return mean_pred, std_pred
    
    def train_batch(self, features_batch: np.ndarray, targets_batch: np.ndarray):
        """Train all models in ensemble"""
        for model in self.models:
            # Add noise to create diversity
            noisy_targets = targets_batch + np.random.normal(0, 0.1, len(targets_batch))
            model.train_batch(features_batch, noisy_targets)


class BanditOptimizer:
    """Multi-armed bandit optimizer for exploration/exploitation"""
    
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.995):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.agent_rewards: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.agent_counts: Dict[str, int] = defaultdict(int)
        self.exploration_bonus = 2.0
    
    def select_agent(self, agent_predictions: Dict[str, ModelPrediction]) -> str:
        """Select agent using epsilon-greedy with UCB"""
        if not agent_predictions:
            return list(agent_predictions.keys())[0]
        
        # Decay epsilon over time
        current_epsilon = self.epsilon * (self.decay_rate ** sum(self.agent_counts.values()))
        
        if np.random.random() < current_epsilon:
            # Exploration: random selection
            return np.random.choice(list(agent_predictions.keys()))
        else:
            # Exploitation with Upper Confidence Bound
            best_agent = None
            best_ucb = float('-inf')
            
            total_counts = sum(self.agent_counts.values())
            
            for agent_id, prediction in agent_predictions.items():
                # Average reward (lower response time is better, so negate)
                avg_reward = -np.mean(self.agent_rewards[agent_id]) if self.agent_rewards[agent_id] else 0
                
                # Confidence interval (UCB)
                if self.agent_counts[agent_id] > 0:
                    confidence = self.exploration_bonus * np.sqrt(
                        np.log(total_counts + 1) / self.agent_counts[agent_id]
                    )
                else:
                    confidence = float('inf')  # Unvisited agents get priority
                
                ucb_value = avg_reward + confidence
                
                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_agent = agent_id
            
            return best_agent or list(agent_predictions.keys())[0]
    
    def update_reward(self, agent_id: str, response_time: float, success: bool):
        """Update agent reward based on performance"""
        # Reward function: lower response time and success is better
        base_reward = 1.0 / (1.0 + response_time) if success else -1.0
        
        self.agent_rewards[agent_id].append(base_reward)
        self.agent_counts[agent_id] += 1


class MLRoutingEngine:
    """
    Machine Learning-powered routing engine.
    
    Features:
    - Learns from historical performance data
    - Predicts agent performance for routing decisions
    - Uses multi-armed bandit for exploration/exploitation
    - Continuously adapts to changing performance patterns
    - Feature engineering from request and agent context
    """
    
    def __init__(
        self,
        model_update_interval: int = 100,  # Update model every N decisions
        training_history_size: int = 10000,
        model_save_path: Optional[str] = None
    ):
        self.model_update_interval = model_update_interval
        self.training_history_size = training_history_size
        self.model_save_path = model_save_path
        
        # ML models per capability
        self.models: Dict[str, EnsembleModel] = {}
        self.bandit_optimizers: Dict[str, BanditOptimizer] = {}
        
        # Training data
        self.training_examples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=training_history_size)
        )
        
        # Performance tracking
        self.decision_count = 0
        self.predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread safety
        self._lock = threading.RLock()
        self._training_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("ML routing engine initialized")
    
    async def start(self):
        """Start the ML routing engine"""
        if self._running:
            logger.warning("ML routing engine already running")
            return
        
        self._running = True
        
        # Load saved models
        await self._load_models()
        
        # Start periodic training
        self._training_task = asyncio.create_task(self._training_loop())
        
        logger.info("ML routing engine started")
    
    async def stop(self):
        """Stop the ML routing engine"""
        self._running = False
        
        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass
        
        # Save models
        await self._save_models()
        
        logger.info("ML routing engine stopped")
    
    async def predict_agent_performance(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> Dict[str, ModelPrediction]:
        """Predict performance for available agents"""
        capability = context.capability or "default"
        predictions = {}
        
        with self._lock:
            # Ensure model exists for this capability
            if capability not in self.models:
                self.models[capability] = EnsembleModel()
                self.bandit_optimizers[capability] = BanditOptimizer()
            
            model = self.models[capability]
            
            for agent in available_agents:
                try:
                    # Extract features
                    features = self._extract_features(context, agent)
                    features_array = self._feature_vector_to_array(features)
                    
                    # Get prediction
                    if model.models[0].trained:
                        pred_response_time, pred_std = model.predict(features_array)
                        
                        # Convert to success probability (heuristic)
                        success_prob = max(0.1, min(0.99, 1.0 / (1.0 + pred_response_time)))
                        confidence = 1.0 / (1.0 + pred_std)
                    else:
                        # Fallback for untrained model
                        pred_response_time = 1.0
                        success_prob = 0.8
                        confidence = 0.5
                    
                    predictions[agent.agent_id] = ModelPrediction(
                        agent_id=agent.agent_id,
                        predicted_response_time=max(0.1, pred_response_time),
                        predicted_success_probability=success_prob,
                        confidence=confidence
                    )
                    
                except Exception as e:
                    logger.error(f"Error predicting for agent {agent.agent_id}: {e}")
                    # Fallback prediction
                    predictions[agent.agent_id] = ModelPrediction(
                        agent_id=agent.agent_id,
                        predicted_response_time=1.0,
                        predicted_success_probability=0.5,
                        confidence=0.1
                    )
        
        return predictions
    
    async def select_optimal_agent(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> Optional[RoutingDecision]:
        """Select optimal agent using ML predictions and bandit optimization"""
        if not available_agents:
            return None
        
        capability = context.capability or "default"
        
        # Get ML predictions
        predictions = await self.predict_agent_performance(context, available_agents)
        
        if not predictions:
            return None
        
        with self._lock:
            # Use bandit optimizer for selection
            if capability not in self.bandit_optimizers:
                self.bandit_optimizers[capability] = BanditOptimizer()
            
            bandit = self.bandit_optimizers[capability]
            selected_agent_id = bandit.select_agent(predictions)
            
            # Find the selected agent
            selected_agent = next(
                (a for a in available_agents if a.agent_id == selected_agent_id),
                available_agents[0]  # Fallback
            )
            
            prediction = predictions[selected_agent_id]
            
            # Store prediction for later evaluation
            self.predictions[capability].append({
                'agent_id': selected_agent_id,
                'prediction': prediction,
                'timestamp': datetime.utcnow(),
                'context': context
            })
            
            self.decision_count += 1
        
        # Create routing decision
        decision = RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.ML_OPTIMIZED,
            decision_factors={
                'predicted_response_time': prediction.predicted_response_time,
                'predicted_success_probability': prediction.predicted_success_probability,
                'model_confidence': prediction.confidence,
                'exploration_rate': bandit.epsilon * (bandit.decay_rate ** sum(bandit.agent_counts.values()))
            },
            confidence_score=prediction.confidence,
            alternative_agents=available_agents[:3]
        )
        
        return decision
    
    async def record_performance(
        self,
        context: RoutingContext,
        agent_id: str,
        actual_response_time: float,
        success: bool
    ):
        """Record actual performance for model training"""
        capability = context.capability or "default"
        
        with self._lock:
            # Update bandit optimizer
            if capability in self.bandit_optimizers:
                self.bandit_optimizers[capability].update_reward(
                    agent_id, actual_response_time, success
                )
            
            # Create training example
            # We need to reconstruct the agent endpoint for feature extraction
            # In practice, this would be stored from the original decision
            agent_endpoint = AgentEndpoint(
                agent_id=agent_id,
                endpoint_url=f"http://{agent_id}:8080",  # Placeholder
                capabilities=[capability]
            )
            
            features = self._extract_features(context, agent_endpoint)
            
            training_example = TrainingExample(
                features=features,
                agent_id=agent_id,
                actual_response_time=actual_response_time,
                success=success,
                timestamp=datetime.utcnow(),
                reward=self._calculate_reward(actual_response_time, success)
            )
            
            self.training_examples[capability].append(training_example)
    
    def _extract_features(self, context: RoutingContext, agent: AgentEndpoint) -> FeatureVector:
        """Extract features from routing context and agent"""
        # Get current time features
        now = datetime.utcnow()
        hour_of_day = now.hour / 24.0
        day_of_week = now.weekday() / 7.0
        
        # Get agent performance data (simplified - would integrate with performance monitor)
        from .performance_monitor import performance_monitor
        agent_stats = performance_monitor.get_agent_stats(agent.agent_id)
        
        if agent_stats:
            agent_response_time = agent_stats.avg_response_time
            agent_success_rate = 1.0 - agent_stats.error_rate
            agent_current_load = agent.current_connections
        else:
            agent_response_time = 1.0
            agent_success_rate = 0.8
            agent_current_load = 0
        
        return FeatureVector(
            request_priority=float(context.priority),
            request_timeout=context.timeout,
            request_size=1.0,  # Would be calculated from actual request
            retry_count=float(context.retry_count),
            
            agent_avg_response_time=agent_response_time,
            agent_success_rate=agent_success_rate,
            agent_current_load=float(agent_current_load),
            agent_capacity_utilization=0.5,  # Would come from system metrics
            agent_recent_errors=0.1,  # Would be calculated from recent history
            
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            recent_request_volume=10.0,  # Would be calculated from recent traffic
            
            agent_performance_trend=1.0,  # Would come from performance analysis
            success_rate_last_hour=agent_success_rate,
            avg_response_time_last_hour=agent_response_time
        )
    
    def _feature_vector_to_array(self, features: FeatureVector) -> np.ndarray:
        """Convert feature vector to numpy array"""
        return np.array([
            features.request_priority,
            features.request_timeout,
            features.request_size,
            features.retry_count,
            features.agent_avg_response_time,
            features.agent_success_rate,
            features.agent_current_load,
            features.agent_capacity_utilization,
            features.agent_recent_errors,
            features.hour_of_day,
            features.day_of_week,
            features.recent_request_volume,
            features.agent_performance_trend
        ])
    
    def _calculate_reward(self, response_time: float, success: bool) -> float:
        """Calculate reward for training"""
        if not success:
            return -1.0
        
        # Reward is inverse of response time (faster is better)
        return 1.0 / (1.0 + response_time)
    
    async def _training_loop(self):
        """Periodic model training loop"""
        while self._running:
            try:
                if self.decision_count >= self.model_update_interval:
                    await self._train_models()
                    self.decision_count = 0
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ML training loop: {e}")
                await asyncio.sleep(10)
    
    async def _train_models(self):
        """Train ML models with accumulated data"""
        with self._lock:
            for capability, examples in self.training_examples.items():
                if len(examples) < 10:  # Need minimum data
                    continue
                
                try:
                    # Prepare training data
                    features_list = []
                    targets_list = []
                    
                    for example in examples:
                        features_array = self._feature_vector_to_array(example.features)
                        target = example.actual_response_time
                        
                        features_list.append(features_array)
                        targets_list.append(target)
                    
                    if not features_list:
                        continue
                    
                    features_batch = np.array(features_list)
                    targets_batch = np.array(targets_list)
                    
                    # Train model
                    if capability not in self.models:
                        self.models[capability] = EnsembleModel()
                    
                    self.models[capability].train_batch(features_batch, targets_batch)
                    
                    logger.info(f"Trained ML model for {capability} with {len(examples)} examples")
                    
                except Exception as e:
                    logger.error(f"Error training model for {capability}: {e}")
    
    async def _save_models(self):
        """Save ML models to disk"""
        if not self.model_save_path:
            return
        
        try:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'bandit_optimizers': self.bandit_optimizers,
                'decision_count': self.decision_count
            }
            
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved ML models to {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def _load_models(self):
        """Load ML models from disk"""
        if not self.model_save_path or not os.path.exists(self.model_save_path):
            return
        
        try:
            with open(self.model_save_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.bandit_optimizers = model_data.get('bandit_optimizers', {})
            self.decision_count = model_data.get('decision_count', 0)
            
            logger.info(f"Loaded ML models from {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_ml_stats(self) -> Dict[str, Any]:
        """Get ML routing statistics"""
        with self._lock:
            stats = {
                'total_decisions': sum(len(examples) for examples in self.training_examples.values()),
                'models_trained': len([m for m in self.models.values() if m.models[0].trained]),
                'capabilities': list(self.models.keys()),
                'decision_count_since_training': self.decision_count
            }
            
            # Per-capability stats
            capability_stats = {}
            for capability in self.models.keys():
                bandit = self.bandit_optimizers.get(capability)
                if bandit:
                    capability_stats[capability] = {
                        'total_selections': sum(bandit.agent_counts.values()),
                        'unique_agents': len(bandit.agent_counts),
                        'exploration_rate': bandit.epsilon * (bandit.decay_rate ** sum(bandit.agent_counts.values()))
                    }
            
            stats['capability_stats'] = capability_stats
            return stats


# Global ML routing engine instance
ml_routing_engine = MLRoutingEngine()