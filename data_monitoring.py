from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass
class DataQualityMetrics:
    timestamp: datetime
    completeness: float  # % of expected data points present
    staleness: float    # max age of data
    anomaly_score: float
    drift_score: float
    
class DataQualityMonitor:
    """
    Monitors data quality metrics and detects anomalies/drift.
    """
    def __init__(
        self,
        expected_frequency: timedelta,
        anomaly_threshold: float = 3.0,
        drift_window: int = 1000
    ):
        self.expected_frequency = expected_frequency
        self.anomaly_threshold = anomaly_threshold
        self.drift_window = drift_window
        
        # Historical metrics for drift detection
        self.metric_history: List[DataQualityMetrics] = []
        # Running statistics for each feature
        self.feature_stats: Dict[str, OnlineNormalizer] = {}
        
    def compute_metrics(
        self,
        measurements: Dict[str, Dict[str, float]],
        timestamp: datetime,
        expected_buoys: List[str]
    ) -> DataQualityMetrics:
        """Compute current data quality metrics."""
        # Completeness
        present_buoys = set(measurements.keys())
        completeness = len(present_buoys) / len(expected_buoys)
        
        # Staleness (assuming measurements include timestamps)
        current_time = datetime.now()
        max_age = max(
            (current_time - timestamp).total_seconds(),
            self.expected_frequency.total_seconds()
        )
        staleness = max_age / self.expected_frequency.total_seconds()
        
        # Anomaly detection
        anomaly_scores = []
        for buoy_id, values in measurements.items():
            for feature, value in values.items():
                key = f"{buoy_id}_{feature}"
                if key not in self.feature_stats:
                    self.feature_stats[key] = OnlineNormalizer()
                normalizer = self.feature_stats[key]
                normalizer.update(value)
                if normalizer.n > 1:  # Need at least 2 points for z-score
                    z_score = abs(normalizer.normalize(value))
                    anomaly_scores.append(z_score)
        
        anomaly_score = max(anomaly_scores) if anomaly_scores else 0.0
        
        # Drift detection using distribution comparison
        drift_score = self._compute_drift_score(measurements)
        
        metrics = DataQualityMetrics(
            timestamp=timestamp,
            completeness=completeness,
            staleness=staleness,
            anomaly_score=anomaly_score,
            drift_score=drift_score
        )
        
        self.metric_history.append(metrics)
        if len(self.metric_history) > self.drift_window:
            self.metric_history.pop(0)
            
        return metrics
    
    def _compute_drift_score(
        self,
        measurements: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute drift score using multiple distribution comparison methods:
        1. KL-divergence approximation for overall distribution
        2. Anderson-Darling test for temporal correlation
        3. Wasserstein distance for feature-wise drift
        """
        if len(self.metric_history) < self.drift_window // 2:
            return 0.0
            
        recent_values = []
        for buoy_data in measurements.values():
            recent_values.extend(buoy_data.values())
            
        historical_values = []
        for metrics in self.metric_history[:-1]:  # Exclude current
            historical_values.extend([
                metrics.completeness,
                metrics.staleness,
                metrics.anomaly_score
            ])
            
        if not recent_values or not historical_values:
            return 0.0

        # 1. KL-divergence for overall distribution
        recent_mean = np.mean(recent_values)
        recent_var = np.var(recent_values) + 1e-6
        hist_mean = np.mean(historical_values)
        hist_var = np.var(historical_values) + 1e-6
        
        kl_score = abs(
            np.log(hist_var / recent_var) +
            (recent_var + (recent_mean - hist_mean)**2) / (2 * hist_var) - 0.5
        )
        
        # 2. Anderson-Darling test for temporal correlation
        from scipy import stats
        _, ad_critical_values, _ = stats.anderson(recent_values, dist='norm')
        ad_score = np.mean([
            abs(stats.anderson(recent_values[i:i+100], dist='norm')[0] / v)
            for i in range(0, len(recent_values)-100, 50)
            for v in [ad_critical_values[2]]  # Using middle critical value
        ] or [0])
        
        # 3. Wasserstein distance for feature-wise comparison
        def wasserstein_1d(u_values, v_values):
            u_values = np.sort(u_values)
            v_values = np.sort(v_values)
            return np.mean(np.abs(u_values - v_values))
        
        w_score = wasserstein_1d(
            np.array(recent_values),
            np.array(historical_values)
        ) / (np.std(historical_values) + 1e-6)  # Normalize by std
        
        # Combine scores with weights
        drift_score = (
            0.4 * kl_score +
            0.3 * min(ad_score, 5.0) +  # Cap AD score
            0.3 * w_score
        )
        
        return drift_score
    
    def get_alerts(self) -> List[str]:
        """Generate alerts based on current metrics."""
        if not self.metric_history:
            return []
            
        current = self.metric_history[-1]
        alerts = []
        
        if current.completeness < 0.8:
            alerts.append(f"Low data completeness: {current.completeness:.2%}")
        
        if current.staleness > 2.0:
            alerts.append(f"Data staleness warning: {current.staleness:.1f}x expected")
            
        if current.anomaly_score > self.anomaly_threshold:
            alerts.append(f"Anomaly detected: score = {current.anomaly_score:.2f}")
            
        if current.drift_score > 1.0:
            alerts.append(f"Significant drift detected: score = {current.drift_score:.2f}")
            
        return alerts