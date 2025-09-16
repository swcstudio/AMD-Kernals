// Rust Language Binding for WebGPU Integration
// Native Rust API for high-performance GPU compute

use std::sync::Arc;
use std::marker::PhantomData;
use std::time::Instant;

use super::{Language, LanguageBinding, TypeConverter, LanguageKernelData, LanguageResult, BindingError, DataType};
use crate::webgpu::WebGPUIntegration;
use crate::core::{ComputeKernel, KernelResult};

/// Rust-specific WebGPU binding
pub struct RustBinding {
    webgpu_integration: Arc<WebGPUIntegration>,
    type_converter: RustTypeConverter,
}

impl RustBinding {
    pub fn new(webgpu_integration: Arc<WebGPUIntegration>) -> Self {
        Self {
            webgpu_integration,
            type_converter: RustTypeConverter::new(),
        }
    }

    /// Execute compute kernel with Rust data types
    pub async fn execute_compute<T: GPUData>(&self,
        data: &[T],
        shader: &str,
        workgroup_size: (u32, u32, u32)
    ) -> Result<Vec<T>, BindingError> {
        let execution_start = Instant::now();

        // Serialize input data
        let input_data = self.type_converter.serialize_data(data)?;

        // Create compute kernel
        let kernel = ComputeKernel {
            compute_shader: shader.to_string(),
            workgroup_size,
            input_data,
            output_layout: self.type_converter.get_output_layout::<T>()?,
            origin: crate::core::Origin::NativeProcess {
                pid: std::process::id(),
                executable_path: std::env::current_exe()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            },
            resource_requirements: crate::webgpu::security::ResourceRequirements::default(),
        };

        // Execute kernel
        let result = self.webgpu_integration.execute_kernel(kernel).await
            .map_err(|e| BindingError::ExecutionError(e.to_string()))?;

        // Deserialize output data
        let output_data = self.type_converter.deserialize_result::<T>(result.output_data)?;

        let execution_time = execution_start.elapsed();
        log::debug!("Rust compute execution completed in {:?}", execution_time);

        Ok(output_data)
    }

    /// Execute compute kernel with custom types and complex layouts
    pub async fn execute_compute_advanced<T: GPUData>(&self,
        input: RustComputeInput<T>,
        shader: &str
    ) -> Result<RustComputeOutput<T>, BindingError> {
        let execution_start = Instant::now();

        // Validate input configuration
        input.validate()?;

        // Prepare compute kernel
        let kernel = ComputeKernel {
            compute_shader: shader.to_string(),
            workgroup_size: input.workgroup_size,
            input_data: self.type_converter.serialize_compute_input(&input)?,
            output_layout: input.output_layout.clone(),
            origin: crate::core::Origin::NativeProcess {
                pid: std::process::id(),
                executable_path: std::env::current_exe()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            },
            resource_requirements: input.resource_requirements.clone(),
        };

        // Execute with performance monitoring
        let result = self.webgpu_integration.execute_kernel(kernel).await
            .map_err(|e| BindingError::ExecutionError(e.to_string()))?;

        // Create output with metadata
        let output = RustComputeOutput {
            data: self.type_converter.deserialize_result::<T>(result.output_data)?,
            execution_time: result.execution_time,
            memory_usage: result.memory_usage,
            performance_metadata: result.performance_metadata,
            execution_start,
        };

        Ok(output)
    }

    /// Create a compute context for multiple operations
    pub async fn create_compute_context(&self) -> Result<RustComputeContext, BindingError> {
        RustComputeContext::new(self.webgpu_integration.clone()).await
    }

    /// Batch execution for multiple kernels
    pub async fn execute_batch<T: GPUData>(&self,
        batch: RustBatchExecution<T>
    ) -> Result<Vec<RustComputeOutput<T>>, BindingError> {
        let mut results = Vec::with_capacity(batch.operations.len());

        for operation in batch.operations {
            let result = self.execute_compute_advanced(operation.input, &operation.shader).await?;
            results.push(result);
        }

        Ok(results)
    }
}

impl LanguageBinding for RustBinding {
    fn get_language(&self) -> Language {
        Language::Rust
    }

    fn get_api_version(&self) -> String {
        "1.0.0".to_string()
    }

    fn is_available(&self) -> bool {
        true // Always available in Rust
    }
}

/// Rust type converter for GPU data serialization
pub struct RustTypeConverter {
    _phantom: PhantomData<()>,
}

impl RustTypeConverter {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Serialize Rust data to GPU-compatible format
    pub fn serialize_data<T: GPUData>(&self, data: &[T]) -> Result<Vec<u8>, BindingError> {
        let mut serialized = Vec::with_capacity(data.len() * std::mem::size_of::<T>());

        for item in data {
            let bytes = item.to_gpu_bytes();
            serialized.extend_from_slice(&bytes);
        }

        Ok(serialized)
    }

    /// Deserialize GPU output to Rust data
    pub fn deserialize_result<T: GPUData>(&self, data: Vec<u8>) -> Result<Vec<T>, BindingError> {
        let item_size = std::mem::size_of::<T>();
        if data.len() % item_size != 0 {
            return Err(BindingError::InvalidDataFormat(
                format!("Data length {} is not divisible by item size {}", data.len(), item_size)
            ));
        }

        let item_count = data.len() / item_size;
        let mut result = Vec::with_capacity(item_count);

        for i in 0..item_count {
            let start = i * item_size;
            let end = start + item_size;
            let item_bytes = &data[start..end];

            let item = T::from_gpu_bytes(item_bytes)
                .map_err(|e| BindingError::DeserializationError(e.to_string()))?;

            result.push(item);
        }

        Ok(result)
    }

    /// Get output layout for type T
    pub fn get_output_layout<T: GPUData>(&self) -> Result<crate::core::OutputLayout, BindingError> {
        Ok(T::get_output_layout())
    }

    /// Serialize complex compute input
    pub fn serialize_compute_input<T: GPUData>(&self, input: &RustComputeInput<T>) -> Result<Vec<u8>, BindingError> {
        let mut serialized = Vec::new();

        // Serialize primary data
        let primary_data = self.serialize_data(&input.data)?;
        serialized.extend_from_slice(&primary_data);

        // Serialize additional buffers
        for buffer in &input.additional_buffers {
            serialized.extend_from_slice(&buffer.data);
        }

        Ok(serialized)
    }
}

impl TypeConverter for RustTypeConverter {
    fn to_compute_kernel(&self, data: LanguageKernelData) -> Result<ComputeKernel, BindingError> {
        match data {
            LanguageKernelData::Rust(rust_data) => {
                Ok(ComputeKernel {
                    compute_shader: rust_data.shader,
                    workgroup_size: rust_data.workgroup_size,
                    input_data: rust_data.input_data,
                    output_layout: rust_data.output_layout,
                    origin: rust_data.origin,
                    resource_requirements: rust_data.resource_requirements,
                })
            },
            _ => Err(BindingError::TypeConversionError(
                "Expected Rust kernel data".to_string()
            )),
        }
    }

    fn from_kernel_result(&self, result: KernelResult) -> Result<LanguageResult, BindingError> {
        Ok(LanguageResult::Rust(RustResult {
            output_data: result.output_data,
            execution_time: result.execution_time,
            memory_usage: result.memory_usage,
            performance_metadata: result.performance_metadata,
        }))
    }

    fn get_supported_types(&self) -> Vec<DataType> {
        vec![
            DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
            DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
            DataType::Float32, DataType::Float64, DataType::Bool,
            DataType::Vec2f32, DataType::Vec3f32, DataType::Vec4f32,
            DataType::Vec2f64, DataType::Vec3f64, DataType::Vec4f64,
            DataType::Mat2x2f32, DataType::Mat3x3f32, DataType::Mat4x4f32,
            DataType::Mat2x2f64, DataType::Mat3x3f64, DataType::Mat4x4f64,
        ]
    }

    fn validate_type_compatibility(&self, data_type: &DataType) -> bool {
        self.get_supported_types().contains(data_type)
    }
}

/// Trait for GPU-compatible Rust data types
pub trait GPUData: Clone + Send + Sync + 'static {
    /// Convert to GPU-compatible byte representation
    fn to_gpu_bytes(&self) -> Vec<u8>;

    /// Create from GPU byte representation
    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String>;

    /// Get the output layout for this type
    fn get_output_layout() -> crate::core::OutputLayout;

    /// Get the GPU data type descriptor
    fn get_data_type() -> DataType;
}

/// Rust compute input configuration
#[derive(Clone)]
pub struct RustComputeInput<T: GPUData> {
    pub data: Vec<T>,
    pub workgroup_size: (u32, u32, u32),
    pub output_layout: crate::core::OutputLayout,
    pub resource_requirements: crate::webgpu::security::ResourceRequirements,
    pub additional_buffers: Vec<RustBuffer>,
    pub compute_parameters: RustComputeParameters,
}

impl<T: GPUData> RustComputeInput<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data,
            workgroup_size: (64, 1, 1),
            output_layout: T::get_output_layout(),
            resource_requirements: crate::webgpu::security::ResourceRequirements::default(),
            additional_buffers: Vec::new(),
            compute_parameters: RustComputeParameters::default(),
        }
    }

    pub fn with_workgroup_size(mut self, size: (u32, u32, u32)) -> Self {
        self.workgroup_size = size;
        self
    }

    pub fn with_output_layout(mut self, layout: crate::core::OutputLayout) -> Self {
        self.output_layout = layout;
        self
    }

    pub fn with_resource_requirements(mut self, requirements: crate::webgpu::security::ResourceRequirements) -> Self {
        self.resource_requirements = requirements;
        self
    }

    pub fn add_buffer(mut self, buffer: RustBuffer) -> Self {
        self.additional_buffers.push(buffer);
        self
    }

    pub fn with_parameters(mut self, parameters: RustComputeParameters) -> Self {
        self.compute_parameters = parameters;
        self
    }

    fn validate(&self) -> Result<(), BindingError> {
        if self.data.is_empty() {
            return Err(BindingError::InvalidDataFormat("Input data cannot be empty".to_string()));
        }

        if self.workgroup_size.0 == 0 || self.workgroup_size.1 == 0 || self.workgroup_size.2 == 0 {
            return Err(BindingError::InvalidDataFormat("Workgroup size dimensions must be greater than 0".to_string()));
        }

        // Validate workgroup size limits
        let total_threads = self.workgroup_size.0 * self.workgroup_size.1 * self.workgroup_size.2;
        if total_threads > 1024 {
            return Err(BindingError::InvalidDataFormat("Total workgroup threads cannot exceed 1024".to_string()));
        }

        Ok(())
    }
}

/// Rust compute output with metadata
#[derive(Clone)]
pub struct RustComputeOutput<T: GPUData> {
    pub data: Vec<T>,
    pub execution_time: std::time::Duration,
    pub memory_usage: usize,
    pub performance_metadata: Option<crate::core::PerformanceMetadata>,
    pub execution_start: Instant,
}

impl<T: GPUData> RustComputeOutput<T> {
    /// Get total execution time including setup
    pub fn total_execution_time(&self) -> std::time::Duration {
        self.execution_start.elapsed()
    }

    /// Get efficiency score if available
    pub fn efficiency_score(&self) -> Option<f64> {
        self.performance_metadata.as_ref().map(|m| m.efficiency_score)
    }

    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        if let Some(metadata) = &self.performance_metadata {
            metadata.memory_usage as f64 / self.memory_usage as f64
        } else {
            0.0
        }
    }
}

/// Additional buffer for complex compute operations
#[derive(Clone)]
pub struct RustBuffer {
    pub data: Vec<u8>,
    pub buffer_type: RustBufferType,
    pub usage: RustBufferUsage,
}

impl RustBuffer {
    pub fn new(data: Vec<u8>, buffer_type: RustBufferType) -> Self {
        Self {
            data,
            buffer_type,
            usage: RustBufferUsage::Storage,
        }
    }

    pub fn uniform_buffer(data: Vec<u8>) -> Self {
        Self {
            data,
            buffer_type: RustBufferType::Uniform,
            usage: RustBufferUsage::Uniform,
        }
    }

    pub fn storage_buffer(data: Vec<u8>) -> Self {
        Self {
            data,
            buffer_type: RustBufferType::Storage,
            usage: RustBufferUsage::Storage,
        }
    }
}

/// Buffer type enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustBufferType {
    Storage,
    Uniform,
    Index,
    Vertex,
}

/// Buffer usage enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustBufferUsage {
    Storage,
    Uniform,
    CopySrc,
    CopyDst,
}

/// Compute parameters for fine-tuning execution
#[derive(Debug, Clone)]
pub struct RustComputeParameters {
    pub optimize_for: OptimizationTarget,
    pub memory_preference: MemoryPreference,
    pub precision_hint: PrecisionHint,
    pub debug_mode: bool,
}

impl Default for RustComputeParameters {
    fn default() -> Self {
        Self {
            optimize_for: OptimizationTarget::Performance,
            memory_preference: MemoryPreference::Balanced,
            precision_hint: PrecisionHint::Default,
            debug_mode: false,
        }
    }
}

/// Optimization target preferences
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationTarget {
    Performance,
    MemoryUsage,
    PowerEfficiency,
    Balanced,
}

/// Memory usage preferences
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPreference {
    MinimalUsage,
    Balanced,
    HighThroughput,
}

/// Precision hint for calculations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionHint {
    Low,      // f16 where possible
    Default,  // f32
    High,     // f64 where needed
}

/// Compute context for managing multiple operations
pub struct RustComputeContext {
    webgpu_integration: Arc<WebGPUIntegration>,
    context_id: uuid::Uuid,
    active_operations: std::sync::Mutex<Vec<uuid::Uuid>>,
}

impl RustComputeContext {
    async fn new(webgpu_integration: Arc<WebGPUIntegration>) -> Result<Self, BindingError> {
        Ok(Self {
            webgpu_integration,
            context_id: uuid::Uuid::new_v4(),
            active_operations: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Execute operation within this context
    pub async fn execute<T: GPUData>(&self,
        input: RustComputeInput<T>,
        shader: &str
    ) -> Result<RustComputeOutput<T>, BindingError> {
        let operation_id = uuid::Uuid::new_v4();

        // Track operation
        self.active_operations.lock().unwrap().push(operation_id);

        // Create binding for execution
        let binding = RustBinding::new(self.webgpu_integration.clone());
        let result = binding.execute_compute_advanced(input, shader).await;

        // Remove from tracking
        self.active_operations.lock().unwrap().retain(|&id| id != operation_id);

        result
    }

    /// Get context statistics
    pub fn get_stats(&self) -> RustContextStats {
        let active_count = self.active_operations.lock().unwrap().len();

        RustContextStats {
            context_id: self.context_id,
            active_operations: active_count,
            created_at: std::time::SystemTime::now(), // Would track actual creation time
        }
    }
}

/// Context statistics
#[derive(Debug, Clone)]
pub struct RustContextStats {
    pub context_id: uuid::Uuid,
    pub active_operations: usize,
    pub created_at: std::time::SystemTime,
}

/// Batch execution configuration
pub struct RustBatchExecution<T: GPUData> {
    pub operations: Vec<RustBatchOperation<T>>,
    pub execution_mode: BatchExecutionMode,
}

/// Individual batch operation
pub struct RustBatchOperation<T: GPUData> {
    pub input: RustComputeInput<T>,
    pub shader: String,
    pub priority: BatchPriority,
}

/// Batch execution modes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchExecutionMode {
    Sequential,
    Parallel,
    Adaptive, // Choose based on system resources
}

/// Batch operation priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchPriority {
    Low,
    Normal,
    High,
    Critical,
}

// Data type definitions for the binding system

/// Rust kernel data for cross-language compatibility
#[derive(Debug, Clone)]
pub struct RustKernelData {
    pub shader: String,
    pub workgroup_size: (u32, u32, u32),
    pub input_data: Vec<u8>,
    pub output_layout: crate::core::OutputLayout,
    pub origin: crate::core::Origin,
    pub resource_requirements: crate::webgpu::security::ResourceRequirements,
}

/// Rust result data for cross-language compatibility
#[derive(Debug, Clone)]
pub struct RustResult {
    pub output_data: Vec<u8>,
    pub execution_time: std::time::Duration,
    pub memory_usage: usize,
    pub performance_metadata: Option<crate::core::PerformanceMetadata>,
}

// Implementations of GPUData for common Rust types

impl GPUData for f32 {
    fn to_gpu_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 4 {
            return Err(format!("Expected 4 bytes for f32, got {}", bytes.len()));
        }
        let array: [u8; 4] = bytes.try_into().map_err(|_| "Invalid byte array")?;
        Ok(f32::from_le_bytes(array))
    }

    fn get_output_layout() -> crate::core::OutputLayout {
        crate::core::OutputLayout::Float32Array(0) // Size determined at runtime
    }

    fn get_data_type() -> DataType {
        DataType::Float32
    }
}

impl GPUData for f64 {
    fn to_gpu_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 8 {
            return Err(format!("Expected 8 bytes for f64, got {}", bytes.len()));
        }
        let array: [u8; 8] = bytes.try_into().map_err(|_| "Invalid byte array")?;
        Ok(f64::from_le_bytes(array))
    }

    fn get_output_layout() -> crate::core::OutputLayout {
        crate::core::OutputLayout::Float64Array(0)
    }

    fn get_data_type() -> DataType {
        DataType::Float64
    }
}

impl GPUData for i32 {
    fn to_gpu_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 4 {
            return Err(format!("Expected 4 bytes for i32, got {}", bytes.len()));
        }
        let array: [u8; 4] = bytes.try_into().map_err(|_| "Invalid byte array")?;
        Ok(i32::from_le_bytes(array))
    }

    fn get_output_layout() -> crate::core::OutputLayout {
        crate::core::OutputLayout::Int32Array(0)
    }

    fn get_data_type() -> DataType {
        DataType::Int32
    }
}

impl GPUData for u32 {
    fn to_gpu_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 4 {
            return Err(format!("Expected 4 bytes for u32, got {}", bytes.len()));
        }
        let array: [u8; 4] = bytes.try_into().map_err(|_| "Invalid byte array")?;
        Ok(u32::from_le_bytes(array))
    }

    fn get_output_layout() -> crate::core::OutputLayout {
        crate::core::OutputLayout::UInt32Array(0)
    }

    fn get_data_type() -> DataType {
        DataType::UInt32
    }
}

// Vector types implementation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3f32 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl GPUData for Vec3f32 {
    fn to_gpu_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&self.x.to_le_bytes());
        bytes.extend_from_slice(&self.y.to_le_bytes());
        bytes.extend_from_slice(&self.z.to_le_bytes());
        bytes
    }

    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 12 {
            return Err(format!("Expected 12 bytes for Vec3f32, got {}", bytes.len()));
        }

        let x = f32::from_le_bytes(bytes[0..4].try_into().map_err(|_| "Invalid x bytes")?);
        let y = f32::from_le_bytes(bytes[4..8].try_into().map_err(|_| "Invalid y bytes")?);
        let z = f32::from_le_bytes(bytes[8..12].try_into().map_err(|_| "Invalid z bytes")?);

        Ok(Vec3f32 { x, y, z })
    }

    fn get_output_layout() -> crate::core::OutputLayout {
        crate::core::OutputLayout::Vec3f32Array(0)
    }

    fn get_data_type() -> DataType {
        DataType::Vec3f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4x4f32 {
    pub data: [f32; 16],
}

impl GPUData for Mat4x4f32 {
    fn to_gpu_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for &value in &self.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn from_gpu_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 64 {
            return Err(format!("Expected 64 bytes for Mat4x4f32, got {}", bytes.len()));
        }

        let mut data = [0.0f32; 16];
        for i in 0..16 {
            let start = i * 4;
            let end = start + 4;
            data[i] = f32::from_le_bytes(
                bytes[start..end].try_into().map_err(|_| format!("Invalid bytes at index {}", i))?
            );
        }

        Ok(Mat4x4f32 { data })
    }

    fn get_output_layout() -> crate::core::OutputLayout {
        crate::core::OutputLayout::Mat4x4f32Array(0)
    }

    fn get_data_type() -> DataType {
        DataType::Mat4x4f32
    }
}