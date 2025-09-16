// WebGPU Adapter Pool - Manages discovery and selection of optimal WebGPU adapters

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::{Adapter, Instance, DeviceType, Backend, Features, Limits};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

use super::{WebGPUError, MemoryError};

/// Pool of discovered WebGPU adapters with performance profiling
pub struct WebGPUAdapterPool {
    adapters: HashMap<AdapterKey, WebGPUAdapter>,
    performance_profiles: HashMap<AdapterKey, PerformanceProfile>,
    compatibility_matrix: CompatibilityMatrix,
    load_balancer: Arc<WebGPULoadBalancer>,
    instance: Arc<Instance>,
    discovery_config: AdapterDiscoveryConfig,
}

impl WebGPUAdapterPool {
    pub fn new(instance: Arc<Instance>) -> Self {
        Self {
            adapters: HashMap::new(),
            performance_profiles: HashMap::new(),
            compatibility_matrix: CompatibilityMatrix::new(),
            load_balancer: Arc::new(WebGPULoadBalancer::new()),
            instance,
            discovery_config: AdapterDiscoveryConfig::default(),
        }
    }

    /// Discover all available WebGPU adapters and benchmark their performance
    pub async fn discover_adapters(&mut self) -> Result<(), WebGPUError> {
        log::info!("Starting WebGPU adapter discovery");
        let discovery_start = Instant::now();

        // Clear existing adapters
        self.adapters.clear();
        self.performance_profiles.clear();

        // Enumerate adapters across all backends
        let adapters = self.instance.enumerate_adapters(wgpu::Backends::all());
        let adapter_count = adapters.len();

        log::info!("Found {} potential WebGPU adapters", adapter_count);

        for (index, adapter) in adapters.enumerate() {
            let info = adapter.get_info();
            let features = adapter.features();
            let limits = adapter.limits();

            log::debug!("Processing adapter {}/{}: {} - {:?}",
                index + 1, adapter_count, info.name, info.backend);

            let adapter_key = AdapterKey {
                vendor_id: info.vendor,
                device_id: info.device,
                backend: info.backend,
                device_type: info.device_type,
            };

            // Validate adapter compatibility
            if !self.is_adapter_compatible(&adapter, &info, &features, &limits) {
                log::warn!("Adapter {} is not compatible, skipping", info.name);
                continue;
            }

            // Benchmark adapter performance
            let performance_profile = match self.benchmark_adapter(&adapter).await {
                Ok(profile) => profile,
                Err(e) => {
                    log::warn!("Failed to benchmark adapter {}: {}", info.name, e);
                    continue;
                }
            };

            // Store adapter and profile
            let webgpu_adapter = WebGPUAdapter::new(adapter, info, features, limits);
            self.adapters.insert(adapter_key, webgpu_adapter);
            self.performance_profiles.insert(adapter_key, performance_profile);

            log::info!("Successfully registered adapter: {}", info.name);
        }

        // Update compatibility matrix
        self.compatibility_matrix.update(&self.adapters, &self.performance_profiles);

        let discovery_time = discovery_start.elapsed();
        log::info!("Adapter discovery completed in {:?}. Registered {} adapters",
            discovery_time, self.adapters.len());

        if self.adapters.is_empty() {
            return Err(WebGPUError::AdapterDiscoveryError(
                "No compatible WebGPU adapters found".to_string()
            ));
        }

        Ok(())
    }

    /// Get the optimal adapter for given backend configuration
    pub fn get_optimal_adapter(&self, config: &BackendConfig) -> Result<&WebGPUAdapter, AdapterError> {
        let candidates = self.filter_compatible_adapters(config)?;

        if candidates.is_empty() {
            return Err(AdapterError::NoCompatibleAdapters);
        }

        let optimal_key = self.select_best_adapter(candidates, config)?;

        self.adapters.get(optimal_key)
            .ok_or(AdapterError::AdapterNotFound(*optimal_key))
    }

    /// Filter adapters based on compatibility requirements
    fn filter_compatible_adapters(&self, config: &BackendConfig) -> Result<Vec<AdapterKey>, AdapterError> {
        let mut candidates = Vec::new();

        for (key, adapter) in &self.adapters {
            if self.is_adapter_suitable_for_config(adapter, config) {
                candidates.push(*key);
            }
        }

        Ok(candidates)
    }

    /// Select the best adapter from candidates based on performance and requirements
    fn select_best_adapter(&self,
        candidates: Vec<AdapterKey>,
        config: &BackendConfig
    ) -> Result<&AdapterKey, AdapterError> {
        let mut best_adapter = None;
        let mut best_score = 0.0f64;

        for key in &candidates {
            let adapter = self.adapters.get(key).unwrap();
            let profile = self.performance_profiles.get(key).unwrap();

            let score = self.calculate_adapter_score(adapter, profile, config);

            if score > best_score {
                best_score = score;
                best_adapter = Some(key);
            }
        }

        best_adapter.ok_or(AdapterError::NoSuitableAdapter)
    }

    /// Calculate fitness score for adapter based on requirements and performance
    fn calculate_adapter_score(&self,
        adapter: &WebGPUAdapter,
        profile: &PerformanceProfile,
        config: &BackendConfig
    ) -> f64 {
        let mut score = 0.0;

        // Base performance score (0-40 points)
        score += profile.compute_performance_score * 40.0;

        // Memory bandwidth score (0-20 points)
        score += profile.memory_bandwidth_score * 20.0;

        // Feature compatibility score (0-15 points)
        let feature_compatibility = self.calculate_feature_compatibility(adapter, config);
        score += feature_compatibility * 15.0;

        // Device type preference (0-10 points)
        score += match adapter.info.device_type {
            DeviceType::DiscreteGpu => 10.0,
            DeviceType::IntegratedGpu => 7.0,
            DeviceType::VirtualGpu => 5.0,
            DeviceType::Cpu => 2.0,
            DeviceType::Other => 1.0,
        };

        // Backend preference (0-10 points)
        score += match adapter.info.backend {
            Backend::Vulkan => 10.0,
            Backend::Metal => 9.0,
            Backend::Dx12 => 8.0,
            Backend::Dx11 => 6.0,
            Backend::Gl => 4.0,
            Backend::BrowserWebGpu => 7.0,
        };

        // Load balancing consideration (0-5 points)
        let load_factor = self.load_balancer.get_adapter_load_factor(&adapter.key());
        score += (1.0 - load_factor) * 5.0;

        score
    }

    /// Calculate feature compatibility score
    fn calculate_feature_compatibility(&self, adapter: &WebGPUAdapter, config: &BackendConfig) -> f64 {
        let required_features = config.required_features.unwrap_or(Features::empty());
        let adapter_features = adapter.features;

        if !adapter_features.contains(required_features) {
            return 0.0; // Hard requirement not met
        }

        // Calculate optional feature coverage
        let optional_features = config.optional_features.unwrap_or(Features::empty());
        let supported_optional = adapter_features & optional_features;

        if optional_features.is_empty() {
            1.0
        } else {
            supported_optional.bits().count_ones() as f64 / optional_features.bits().count_ones() as f64
        }
    }

    /// Check if adapter meets basic compatibility requirements
    fn is_adapter_compatible(&self,
        adapter: &Adapter,
        info: &wgpu::AdapterInfo,
        features: &Features,
        limits: &Limits
    ) -> bool {
        // Check minimum compute capability
        if !features.contains(Features::SHADER_F16) {
            log::debug!("Adapter {} lacks F16 shader support", info.name);
        }

        // Check minimum limits
        if limits.max_compute_workgroup_size_x < self.discovery_config.min_workgroup_size {
            log::debug!("Adapter {} has insufficient workgroup size", info.name);
            return false;
        }

        if limits.max_compute_invocations_per_workgroup < self.discovery_config.min_invocations_per_workgroup {
            log::debug!("Adapter {} has insufficient invocations per workgroup", info.name);
            return false;
        }

        true
    }

    /// Check if adapter is suitable for specific backend configuration
    fn is_adapter_suitable_for_config(&self, adapter: &WebGPUAdapter, config: &BackendConfig) -> bool {
        // Check required features
        if let Some(required_features) = config.required_features {
            if !adapter.features.contains(required_features) {
                return false;
            }
        }

        // Check limits
        if let Some(min_workgroup_size) = config.min_workgroup_size {
            if adapter.limits.max_compute_workgroup_size_x < min_workgroup_size {
                return false;
            }
        }

        // Check device type preference
        if let Some(preferred_device_types) = &config.preferred_device_types {
            if !preferred_device_types.contains(&adapter.info.device_type) {
                return false;
            }
        }

        true
    }

    /// Benchmark adapter performance across various workloads
    async fn benchmark_adapter(&self, adapter: &Adapter) -> Result<PerformanceProfile, WebGPUError> {
        log::debug!("Benchmarking adapter: {}", adapter.get_info().name);

        let device = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Benchmark Device"),
                required_features: Features::TIMESTAMP_QUERY | Features::PIPELINE_STATISTICS_QUERY,
                required_limits: Limits::default(),
            },
            None
        ).await.map_err(|e| WebGPUError::DeviceCreationError(e.to_string()))?;

        let benchmarks = vec![
            self.benchmark_compute_performance(&device).await?,
            self.benchmark_memory_bandwidth(&device).await?,
            self.benchmark_latency(&device).await?,
        ];

        Ok(PerformanceProfile::from_benchmarks(benchmarks))
    }

    /// Benchmark compute performance with matrix multiplication
    async fn benchmark_compute_performance(&self, device: &wgpu::Device) -> Result<BenchmarkResult, WebGPUError> {
        let shader_source = include_str!("shaders/matrix_multiply_benchmark.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Benchmark Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let start_time = Instant::now();

        // Execute matrix multiplication benchmark
        // Implementation details...

        let execution_time = start_time.elapsed();
        let operations_per_second = self.calculate_ops_per_second(execution_time);

        Ok(BenchmarkResult {
            benchmark_type: BenchmarkType::ComputePerformance,
            execution_time,
            operations_per_second,
            memory_usage: 0, // Calculate actual memory usage
        })
    }

    /// Benchmark memory bandwidth with memory-intensive operations
    async fn benchmark_memory_bandwidth(&self, device: &wgpu::Device) -> Result<BenchmarkResult, WebGPUError> {
        let shader_source = include_str!("shaders/memory_bandwidth_benchmark.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Memory Benchmark Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let start_time = Instant::now();

        // Execute memory bandwidth benchmark
        // Implementation details...

        let execution_time = start_time.elapsed();
        let bandwidth_gbps = self.calculate_memory_bandwidth(execution_time);

        Ok(BenchmarkResult {
            benchmark_type: BenchmarkType::MemoryBandwidth,
            execution_time,
            operations_per_second: bandwidth_gbps,
            memory_usage: 0, // Calculate actual memory usage
        })
    }

    /// Benchmark execution latency
    async fn benchmark_latency(&self, device: &wgpu::Device) -> Result<BenchmarkResult, WebGPUError> {
        let shader_source = include_str!("shaders/latency_benchmark.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Latency Benchmark Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let start_time = Instant::now();

        // Execute minimal kernel to measure latency
        // Implementation details...

        let execution_time = start_time.elapsed();

        Ok(BenchmarkResult {
            benchmark_type: BenchmarkType::Latency,
            execution_time,
            operations_per_second: 0.0,
            memory_usage: 0,
        })
    }

    fn calculate_ops_per_second(&self, execution_time: Duration) -> f64 {
        // Calculate operations per second based on known benchmark workload
        let total_operations = 1_000_000.0; // Example: 1M operations
        total_operations / execution_time.as_secs_f64()
    }

    fn calculate_memory_bandwidth(&self, execution_time: Duration) -> f64 {
        // Calculate memory bandwidth in GB/s
        let total_bytes = 1_000_000_000.0; // Example: 1GB transferred
        (total_bytes / execution_time.as_secs_f64()) / 1_000_000_000.0
    }
}

/// Wrapper for WebGPU adapter with additional metadata
#[derive(Clone)]
pub struct WebGPUAdapter {
    adapter: Arc<Adapter>,
    pub info: wgpu::AdapterInfo,
    pub features: Features,
    pub limits: Limits,
    registration_time: Instant,
}

impl WebGPUAdapter {
    pub fn new(adapter: Adapter, info: wgpu::AdapterInfo, features: Features, limits: Limits) -> Self {
        Self {
            adapter: Arc::new(adapter),
            info,
            features,
            limits,
            registration_time: Instant::now(),
        }
    }

    pub fn key(&self) -> AdapterKey {
        AdapterKey {
            vendor_id: self.info.vendor,
            device_id: self.info.device,
            backend: self.info.backend,
            device_type: self.info.device_type,
        }
    }

    pub async fn request_device(&self, descriptor: &wgpu::DeviceDescriptor) -> Result<wgpu::Device, wgpu::RequestDeviceError> {
        self.adapter.request_device(descriptor, None).await
    }

    pub fn get_adapter(&self) -> &Adapter {
        &self.adapter
    }
}

/// Unique identifier for WebGPU adapters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdapterKey {
    pub vendor_id: u32,
    pub device_id: u32,
    pub backend: Backend,
    pub device_type: DeviceType,
}

/// Performance profile for an adapter based on benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub compute_performance_score: f64,    // 0.0 - 1.0
    pub memory_bandwidth_score: f64,       // 0.0 - 1.0
    pub latency_score: f64,               // 0.0 - 1.0 (lower latency = higher score)
    pub overall_score: f64,               // Weighted average
    pub benchmarks: Vec<BenchmarkResult>,
    pub created_at: Instant,
}

impl PerformanceProfile {
    pub fn from_benchmarks(benchmarks: Vec<BenchmarkResult>) -> Self {
        let mut compute_score = 0.0;
        let mut memory_score = 0.0;
        let mut latency_score = 0.0;

        for benchmark in &benchmarks {
            match benchmark.benchmark_type {
                BenchmarkType::ComputePerformance => {
                    compute_score = (benchmark.operations_per_second / 1_000_000.0).min(1.0);
                },
                BenchmarkType::MemoryBandwidth => {
                    memory_score = (benchmark.operations_per_second / 100.0).min(1.0); // 100 GB/s max
                },
                BenchmarkType::Latency => {
                    let latency_ms = benchmark.execution_time.as_millis() as f64;
                    latency_score = (10.0 / latency_ms).min(1.0); // 10ms baseline
                },
            }
        }

        let overall_score = (compute_score * 0.5) + (memory_score * 0.3) + (latency_score * 0.2);

        Self {
            compute_performance_score: compute_score,
            memory_bandwidth_score: memory_score,
            latency_score,
            overall_score,
            benchmarks,
            created_at: Instant::now(),
        }
    }
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_type: BenchmarkType,
    pub execution_time: Duration,
    pub operations_per_second: f64,
    pub memory_usage: usize,
}

/// Types of benchmarks performed
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BenchmarkType {
    ComputePerformance,
    MemoryBandwidth,
    Latency,
}

/// Compatibility matrix for adapter selection
pub struct CompatibilityMatrix {
    feature_requirements: HashMap<String, Features>,
    limit_requirements: HashMap<String, LimitRequirements>,
    backend_preferences: Vec<Backend>,
}

impl CompatibilityMatrix {
    pub fn new() -> Self {
        Self {
            feature_requirements: HashMap::new(),
            limit_requirements: HashMap::new(),
            backend_preferences: vec![
                Backend::Vulkan,
                Backend::Metal,
                Backend::Dx12,
                Backend::BrowserWebGpu,
                Backend::Dx11,
                Backend::Gl,
            ],
        }
    }

    pub fn update(&mut self,
        adapters: &HashMap<AdapterKey, WebGPUAdapter>,
        profiles: &HashMap<AdapterKey, PerformanceProfile>
    ) {
        // Update compatibility matrix based on discovered adapters
        log::debug!("Updating compatibility matrix with {} adapters", adapters.len());
    }
}

/// Load balancer for distributing work across adapters
pub struct WebGPULoadBalancer {
    adapter_loads: Arc<RwLock<HashMap<AdapterKey, AdapterLoad>>>,
}

impl WebGPULoadBalancer {
    pub fn new() -> Self {
        Self {
            adapter_loads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn get_adapter_load_factor(&self, key: &AdapterKey) -> f64 {
        // Return current load factor (0.0 = no load, 1.0 = fully loaded)
        0.0 // Simplified implementation
    }
}

/// Current load information for an adapter
#[derive(Debug, Clone)]
pub struct AdapterLoad {
    pub active_contexts: usize,
    pub pending_operations: usize,
    pub memory_utilization: f64,
    pub last_updated: Instant,
}

/// Configuration for adapter discovery
#[derive(Debug, Clone)]
pub struct AdapterDiscoveryConfig {
    pub min_workgroup_size: u32,
    pub min_invocations_per_workgroup: u32,
    pub required_features: Features,
    pub benchmark_timeout: Duration,
}

impl Default for AdapterDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_workgroup_size: 64,
            min_invocations_per_workgroup: 256,
            required_features: Features::empty(),
            benchmark_timeout: Duration::from_secs(30),
        }
    }
}

/// Backend configuration for adapter selection
#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub required_features: Option<Features>,
    pub optional_features: Option<Features>,
    pub min_workgroup_size: Option<u32>,
    pub max_workgroup_size: Option<u32>,
    pub max_invocations_per_workgroup: Option<u32>,
    pub max_workgroup_storage: Option<u32>,
    pub preferred_device_types: Option<Vec<DeviceType>>,
    pub preferred_backends: Option<Vec<Backend>>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            required_features: None,
            optional_features: None,
            min_workgroup_size: None,
            max_workgroup_size: None,
            max_invocations_per_workgroup: None,
            max_workgroup_storage: None,
            preferred_device_types: None,
            preferred_backends: None,
        }
    }
}

/// Limit requirements for compatibility checking
#[derive(Debug, Clone)]
pub struct LimitRequirements {
    pub min_workgroup_size_x: u32,
    pub min_workgroup_size_y: u32,
    pub min_workgroup_size_z: u32,
    pub min_invocations_per_workgroup: u32,
    pub min_workgroup_storage_size: u32,
}

/// Error types for adapter operations
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("No compatible adapters found")]
    NoCompatibleAdapters,

    #[error("No suitable adapter found for configuration")]
    NoSuitableAdapter,

    #[error("Adapter not found: {0:?}")]
    AdapterNotFound(AdapterKey),

    #[error("Adapter discovery failed: {0}")]
    DiscoveryFailed(String),

    #[error("Benchmarking failed: {0}")]
    BenchmarkingFailed(String),
}