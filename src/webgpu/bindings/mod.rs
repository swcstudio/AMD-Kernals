// WebGPU Multi-Language Bindings
// Cross-language integration for Rust, Elixir, Julia, Zig, and Nim

pub mod rust;
pub mod elixir;
pub mod julia;
pub mod zig;
pub mod nim;

use std::sync::Arc;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::webgpu::WebGPUIntegration;
use crate::core::{ComputeKernel, KernelResult};

/// Multi-language binding coordinator
pub struct MultiLanguageBindings {
    webgpu_integration: Arc<WebGPUIntegration>,
    type_converters: HashMap<Language, Box<dyn TypeConverter>>,
    execution_context: Arc<BindingExecutionContext>,
}

impl MultiLanguageBindings {
    pub async fn new(webgpu_integration: Arc<WebGPUIntegration>) -> Result<Self, BindingError> {
        let mut type_converters: HashMap<Language, Box<dyn TypeConverter>> = HashMap::new();

        // Initialize type converters for each language
        type_converters.insert(Language::Rust, Box::new(rust::RustTypeConverter::new()));
        type_converters.insert(Language::Elixir, Box::new(elixir::ElixirTypeConverter::new()));
        type_converters.insert(Language::Julia, Box::new(julia::JuliaTypeConverter::new()));
        type_converters.insert(Language::Zig, Box::new(zig::ZigTypeConverter::new()));
        type_converters.insert(Language::Nim, Box::new(nim::NimTypeConverter::new()));

        Ok(Self {
            webgpu_integration,
            type_converters,
            execution_context: Arc::new(BindingExecutionContext::new()),
        })
    }

    /// Execute kernel from any supported language
    pub async fn execute_cross_language_kernel(&self,
        language: Language,
        kernel_data: LanguageKernelData
    ) -> Result<LanguageResult, BindingError> {
        // Get appropriate type converter
        let converter = self.type_converters.get(&language)
            .ok_or(BindingError::UnsupportedLanguage(language))?;

        // Convert language-specific data to internal format
        let compute_kernel = converter.to_compute_kernel(kernel_data)?;

        // Execute kernel
        let result = self.webgpu_integration.execute_kernel(compute_kernel).await
            .map_err(|e| BindingError::ExecutionError(e.to_string()))?;

        // Convert result back to language-specific format
        let language_result = converter.from_kernel_result(result)?;

        Ok(language_result)
    }

    /// Get binding for specific language
    pub fn get_language_binding(&self, language: Language) -> Result<&dyn LanguageBinding, BindingError> {
        match language {
            Language::Rust => Ok(&rust::RustBinding::new(self.webgpu_integration.clone())),
            Language::Elixir => Ok(&elixir::ElixirBinding::new(self.webgpu_integration.clone())),
            Language::Julia => Ok(&julia::JuliaBinding::new(self.webgpu_integration.clone())),
            Language::Zig => Ok(&zig::ZigBinding::new(self.webgpu_integration.clone())),
            Language::Nim => Ok(&nim::NimBinding::new(self.webgpu_integration.clone())),
        }
    }
}

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Rust,
    Elixir,
    Julia,
    Zig,
    Nim,
}

/// Language-agnostic kernel data
#[derive(Debug, Clone)]
pub enum LanguageKernelData {
    Rust(rust::RustKernelData),
    Elixir(elixir::ElixirKernelData),
    Julia(julia::JuliaKernelData),
    Zig(zig::ZigKernelData),
    Nim(nim::NimKernelData),
}

/// Language-agnostic result data
#[derive(Debug, Clone)]
pub enum LanguageResult {
    Rust(rust::RustResult),
    Elixir(elixir::ElixirResult),
    Julia(julia::JuliaResult),
    Zig(zig::ZigResult),
    Nim(nim::NimResult),
}

/// Type converter trait for language-specific data transformation
pub trait TypeConverter: Send + Sync {
    fn to_compute_kernel(&self, data: LanguageKernelData) -> Result<ComputeKernel, BindingError>;
    fn from_kernel_result(&self, result: KernelResult) -> Result<LanguageResult, BindingError>;
    fn get_supported_types(&self) -> Vec<DataType>;
    fn validate_type_compatibility(&self, data_type: &DataType) -> bool;
}

/// Language binding trait for language-specific interfaces
pub trait LanguageBinding: Send + Sync {
    fn get_language(&self) -> Language;
    fn get_api_version(&self) -> String;
    fn is_available(&self) -> bool;
}

/// Execution context for bindings
pub struct BindingExecutionContext {
    active_executions: std::sync::Mutex<HashMap<String, ExecutionInfo>>,
    performance_tracker: Arc<BindingPerformanceTracker>,
}

impl BindingExecutionContext {
    pub fn new() -> Self {
        Self {
            active_executions: std::sync::Mutex::new(HashMap::new()),
            performance_tracker: Arc::new(BindingPerformanceTracker::new()),
        }
    }

    pub fn track_execution(&self, execution_id: String, info: ExecutionInfo) {
        self.active_executions.lock().unwrap().insert(execution_id, info);
    }

    pub fn complete_execution(&self, execution_id: &str) -> Option<ExecutionInfo> {
        self.active_executions.lock().unwrap().remove(execution_id)
    }
}

/// Execution tracking information
#[derive(Debug, Clone)]
pub struct ExecutionInfo {
    pub language: Language,
    pub start_time: std::time::Instant,
    pub data_size: usize,
    pub operation_type: String,
}

/// Performance tracking for language bindings
pub struct BindingPerformanceTracker {
    language_metrics: std::sync::Mutex<HashMap<Language, LanguageMetrics>>,
}

impl BindingPerformanceTracker {
    pub fn new() -> Self {
        Self {
            language_metrics: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub fn record_execution(&self, language: Language, duration: std::time::Duration, data_size: usize) {
        let mut metrics = self.language_metrics.lock().unwrap();
        let lang_metrics = metrics.entry(language).or_insert_with(LanguageMetrics::new);
        lang_metrics.record_execution(duration, data_size);
    }

    pub fn get_metrics(&self, language: Language) -> Option<LanguageMetrics> {
        self.language_metrics.lock().unwrap().get(&language).cloned()
    }
}

/// Performance metrics per language
#[derive(Debug, Clone)]
pub struct LanguageMetrics {
    pub total_executions: u64,
    pub total_duration: std::time::Duration,
    pub average_duration: std::time::Duration,
    pub total_data_processed: usize,
    pub throughput_mbps: f64,
}

impl LanguageMetrics {
    pub fn new() -> Self {
        Self {
            total_executions: 0,
            total_duration: std::time::Duration::from_secs(0),
            average_duration: std::time::Duration::from_secs(0),
            total_data_processed: 0,
            throughput_mbps: 0.0,
        }
    }

    pub fn record_execution(&mut self, duration: std::time::Duration, data_size: usize) {
        self.total_executions += 1;
        self.total_duration += duration;
        self.average_duration = self.total_duration / self.total_executions as u32;
        self.total_data_processed += data_size;

        // Calculate throughput in MB/s
        let total_seconds = self.total_duration.as_secs_f64();
        if total_seconds > 0.0 {
            self.throughput_mbps = (self.total_data_processed as f64 / 1_000_000.0) / total_seconds;
        }
    }
}

/// Supported data types across languages
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    // Primitive types
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,

    // Vector types
    Vec2f32,
    Vec3f32,
    Vec4f32,
    Vec2f64,
    Vec3f64,
    Vec4f64,
    Vec2i32,
    Vec3i32,
    Vec4i32,

    // Matrix types
    Mat2x2f32,
    Mat3x3f32,
    Mat4x4f32,
    Mat2x2f64,
    Mat3x3f64,
    Mat4x4f64,

    // Array types
    Array(Box<DataType>, Option<usize>), // (element_type, optional_size)

    // Complex types
    Struct(Vec<StructField>),
    Union(Vec<DataType>),

    // Language-specific types
    RustSpecific(String),
    ElixirSpecific(String),
    JuliaSpecific(String),
    ZigSpecific(String),
    NimSpecific(String),
}

/// Struct field definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub data_type: DataType,
    pub offset: usize,
}

/// Type mapping utilities
pub struct TypeMapper {
    mappings: HashMap<(Language, String), DataType>,
}

impl TypeMapper {
    pub fn new() -> Self {
        let mut mapper = Self {
            mappings: HashMap::new(),
        };
        mapper.initialize_mappings();
        mapper
    }

    fn initialize_mappings(&mut self) {
        // Rust type mappings
        self.mappings.insert((Language::Rust, "i8".to_string()), DataType::Int8);
        self.mappings.insert((Language::Rust, "i16".to_string()), DataType::Int16);
        self.mappings.insert((Language::Rust, "i32".to_string()), DataType::Int32);
        self.mappings.insert((Language::Rust, "i64".to_string()), DataType::Int64);
        self.mappings.insert((Language::Rust, "u8".to_string()), DataType::UInt8);
        self.mappings.insert((Language::Rust, "u16".to_string()), DataType::UInt16);
        self.mappings.insert((Language::Rust, "u32".to_string()), DataType::UInt32);
        self.mappings.insert((Language::Rust, "u64".to_string()), DataType::UInt64);
        self.mappings.insert((Language::Rust, "f32".to_string()), DataType::Float32);
        self.mappings.insert((Language::Rust, "f64".to_string()), DataType::Float64);
        self.mappings.insert((Language::Rust, "bool".to_string()), DataType::Bool);

        // Elixir type mappings
        self.mappings.insert((Language::Elixir, "integer".to_string()), DataType::Int64);
        self.mappings.insert((Language::Elixir, "float".to_string()), DataType::Float64);
        self.mappings.insert((Language::Elixir, "boolean".to_string()), DataType::Bool);

        // Julia type mappings
        self.mappings.insert((Language::Julia, "Int8".to_string()), DataType::Int8);
        self.mappings.insert((Language::Julia, "Int16".to_string()), DataType::Int16);
        self.mappings.insert((Language::Julia, "Int32".to_string()), DataType::Int32);
        self.mappings.insert((Language::Julia, "Int64".to_string()), DataType::Int64);
        self.mappings.insert((Language::Julia, "UInt8".to_string()), DataType::UInt8);
        self.mappings.insert((Language::Julia, "UInt16".to_string()), DataType::UInt16);
        self.mappings.insert((Language::Julia, "UInt32".to_string()), DataType::UInt32);
        self.mappings.insert((Language::Julia, "UInt64".to_string()), DataType::UInt64);
        self.mappings.insert((Language::Julia, "Float32".to_string()), DataType::Float32);
        self.mappings.insert((Language::Julia, "Float64".to_string()), DataType::Float64);
        self.mappings.insert((Language::Julia, "Bool".to_string()), DataType::Bool);

        // Zig type mappings
        self.mappings.insert((Language::Zig, "i8".to_string()), DataType::Int8);
        self.mappings.insert((Language::Zig, "i16".to_string()), DataType::Int16);
        self.mappings.insert((Language::Zig, "i32".to_string()), DataType::Int32);
        self.mappings.insert((Language::Zig, "i64".to_string()), DataType::Int64);
        self.mappings.insert((Language::Zig, "u8".to_string()), DataType::UInt8);
        self.mappings.insert((Language::Zig, "u16".to_string()), DataType::UInt16);
        self.mappings.insert((Language::Zig, "u32".to_string()), DataType::UInt32);
        self.mappings.insert((Language::Zig, "u64".to_string()), DataType::UInt64);
        self.mappings.insert((Language::Zig, "f32".to_string()), DataType::Float32);
        self.mappings.insert((Language::Zig, "f64".to_string()), DataType::Float64);
        self.mappings.insert((Language::Zig, "bool".to_string()), DataType::Bool);

        // Nim type mappings
        self.mappings.insert((Language::Nim, "int8".to_string()), DataType::Int8);
        self.mappings.insert((Language::Nim, "int16".to_string()), DataType::Int16);
        self.mappings.insert((Language::Nim, "int32".to_string()), DataType::Int32);
        self.mappings.insert((Language::Nim, "int64".to_string()), DataType::Int64);
        self.mappings.insert((Language::Nim, "uint8".to_string()), DataType::UInt8);
        self.mappings.insert((Language::Nim, "uint16".to_string()), DataType::UInt16);
        self.mappings.insert((Language::Nim, "uint32".to_string()), DataType::UInt32);
        self.mappings.insert((Language::Nim, "uint64".to_string()), DataType::UInt64);
        self.mappings.insert((Language::Nim, "float32".to_string()), DataType::Float32);
        self.mappings.insert((Language::Nim, "float64".to_string()), DataType::Float64);
        self.mappings.insert((Language::Nim, "bool".to_string()), DataType::Bool);
    }

    pub fn map_type(&self, language: Language, type_name: &str) -> Option<&DataType> {
        self.mappings.get(&(language, type_name.to_string()))
    }

    pub fn reverse_map_type(&self, language: Language, data_type: &DataType) -> Option<String> {
        for ((lang, type_name), mapped_type) in &self.mappings {
            if *lang == language && mapped_type == data_type {
                return Some(type_name.clone());
            }
        }
        None
    }
}

/// Cross-language data serialization utilities
pub struct CrossLanguageSerializer {
    type_mapper: TypeMapper,
}

impl CrossLanguageSerializer {
    pub fn new() -> Self {
        Self {
            type_mapper: TypeMapper::new(),
        }
    }

    /// Serialize data from source language to universal format
    pub fn serialize_data(&self,
        source_language: Language,
        data: &[u8],
        data_type: &DataType
    ) -> Result<Vec<u8>, BindingError> {
        match source_language {
            Language::Rust => self.serialize_rust_data(data, data_type),
            Language::Elixir => self.serialize_elixir_data(data, data_type),
            Language::Julia => self.serialize_julia_data(data, data_type),
            Language::Zig => self.serialize_zig_data(data, data_type),
            Language::Nim => self.serialize_nim_data(data, data_type),
        }
    }

    /// Deserialize data from universal format to target language
    pub fn deserialize_data(&self,
        target_language: Language,
        data: &[u8],
        data_type: &DataType
    ) -> Result<Vec<u8>, BindingError> {
        match target_language {
            Language::Rust => self.deserialize_to_rust_data(data, data_type),
            Language::Elixir => self.deserialize_to_elixir_data(data, data_type),
            Language::Julia => self.deserialize_to_julia_data(data, data_type),
            Language::Zig => self.deserialize_to_zig_data(data, data_type),
            Language::Nim => self.deserialize_to_nim_data(data, data_type),
        }
    }

    fn serialize_rust_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Rust data is already in the expected binary format for most cases
        Ok(data.to_vec())
    }

    fn serialize_elixir_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Handle Elixir-specific serialization (e.g., ETF to binary)
        // This would involve parsing Elixir Term Format (ETF) if needed
        Ok(data.to_vec())
    }

    fn serialize_julia_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Handle Julia-specific serialization
        // Julia arrays are typically column-major, may need transposition
        match data_type {
            DataType::Array(element_type, _) => {
                // Handle array layout conversion if needed
                Ok(data.to_vec())
            },
            _ => Ok(data.to_vec())
        }
    }

    fn serialize_zig_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Handle Zig-specific serialization
        Ok(data.to_vec())
    }

    fn serialize_nim_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Handle Nim-specific serialization
        Ok(data.to_vec())
    }

    fn deserialize_to_rust_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        Ok(data.to_vec())
    }

    fn deserialize_to_elixir_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Convert to Elixir-compatible format
        Ok(data.to_vec())
    }

    fn deserialize_to_julia_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        // Convert to Julia-compatible format (column-major if needed)
        Ok(data.to_vec())
    }

    fn deserialize_to_zig_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        Ok(data.to_vec())
    }

    fn deserialize_to_nim_data(&self, data: &[u8], data_type: &DataType) -> Result<Vec<u8>, BindingError> {
        Ok(data.to_vec())
    }
}

/// Error types for binding operations
#[derive(Debug, thiserror::Error)]
pub enum BindingError {
    #[error("Unsupported language: {0:?}")]
    UnsupportedLanguage(Language),

    #[error("Type conversion error: {0}")]
    TypeConversionError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),

    #[error("Language binding not available: {0:?}")]
    BindingNotAvailable(Language),
}

/// Memory layout utilities for cross-language compatibility
pub struct MemoryLayoutConverter {
    alignment_requirements: HashMap<DataType, usize>,
}

impl MemoryLayoutConverter {
    pub fn new() -> Self {
        let mut alignment_requirements = HashMap::new();

        // Set standard alignment requirements
        alignment_requirements.insert(DataType::Int8, 1);
        alignment_requirements.insert(DataType::Int16, 2);
        alignment_requirements.insert(DataType::Int32, 4);
        alignment_requirements.insert(DataType::Int64, 8);
        alignment_requirements.insert(DataType::UInt8, 1);
        alignment_requirements.insert(DataType::UInt16, 2);
        alignment_requirements.insert(DataType::UInt32, 4);
        alignment_requirements.insert(DataType::UInt64, 8);
        alignment_requirements.insert(DataType::Float32, 4);
        alignment_requirements.insert(DataType::Float64, 8);
        alignment_requirements.insert(DataType::Bool, 1);

        Self { alignment_requirements }
    }

    pub fn calculate_struct_layout(&self, fields: &[StructField]) -> Result<StructLayout, BindingError> {
        let mut offset = 0;
        let mut field_offsets = Vec::new();
        let mut max_alignment = 1;

        for field in fields {
            let alignment = self.get_alignment(&field.data_type)?;
            max_alignment = max_alignment.max(alignment);

            // Align offset to field alignment
            offset = (offset + alignment - 1) & !(alignment - 1);
            field_offsets.push(offset);

            let size = self.get_size(&field.data_type)?;
            offset += size;
        }

        // Align total size to struct alignment
        let total_size = (offset + max_alignment - 1) & !(max_alignment - 1);

        Ok(StructLayout {
            total_size,
            alignment: max_alignment,
            field_offsets,
        })
    }

    fn get_alignment(&self, data_type: &DataType) -> Result<usize, BindingError> {
        self.alignment_requirements.get(data_type)
            .copied()
            .ok_or_else(|| BindingError::TypeConversionError(
                format!("Unknown alignment for type: {:?}", data_type)
            ))
    }

    fn get_size(&self, data_type: &DataType) -> Result<usize, BindingError> {
        match data_type {
            DataType::Int8 | DataType::UInt8 | DataType::Bool => Ok(1),
            DataType::Int16 | DataType::UInt16 => Ok(2),
            DataType::Int32 | DataType::UInt32 | DataType::Float32 => Ok(4),
            DataType::Int64 | DataType::UInt64 | DataType::Float64 => Ok(8),
            DataType::Vec2f32 => Ok(8),
            DataType::Vec3f32 => Ok(12),
            DataType::Vec4f32 => Ok(16),
            DataType::Vec2f64 => Ok(16),
            DataType::Vec3f64 => Ok(24),
            DataType::Vec4f64 => Ok(32),
            DataType::Mat2x2f32 => Ok(16),
            DataType::Mat3x3f32 => Ok(36),
            DataType::Mat4x4f32 => Ok(64),
            DataType::Mat2x2f64 => Ok(32),
            DataType::Mat3x3f64 => Ok(72),
            DataType::Mat4x4f64 => Ok(128),
            DataType::Array(element_type, Some(size)) => {
                let element_size = self.get_size(element_type)?;
                Ok(element_size * size)
            },
            DataType::Struct(fields) => {
                let layout = self.calculate_struct_layout(fields)?;
                Ok(layout.total_size)
            },
            _ => Err(BindingError::TypeConversionError(
                format!("Cannot determine size for type: {:?}", data_type)
            )),
        }
    }
}

/// Struct memory layout information
#[derive(Debug, Clone)]
pub struct StructLayout {
    pub total_size: usize,
    pub alignment: usize,
    pub field_offsets: Vec<usize>,
}

/// Cross-language validation utilities
pub struct CrossLanguageValidator {
    type_mapper: TypeMapper,
}

impl CrossLanguageValidator {
    pub fn new() -> Self {
        Self {
            type_mapper: TypeMapper::new(),
        }
    }

    /// Validate that a type conversion is safe between languages
    pub fn validate_type_conversion(&self,
        source_lang: Language,
        target_lang: Language,
        data_type: &DataType
    ) -> Result<(), BindingError> {
        // Check if both languages support the data type
        let source_supports = self.language_supports_type(source_lang, data_type);
        let target_supports = self.language_supports_type(target_lang, data_type);

        if !source_supports {
            return Err(BindingError::TypeConversionError(
                format!("{:?} does not support type {:?}", source_lang, data_type)
            ));
        }

        if !target_supports {
            return Err(BindingError::TypeConversionError(
                format!("{:?} does not support type {:?}", target_lang, data_type)
            ));
        }

        // Check for specific conversion issues
        self.check_specific_conversion_issues(source_lang, target_lang, data_type)?;

        Ok(())
    }

    fn language_supports_type(&self, language: Language, data_type: &DataType) -> bool {
        match (language, data_type) {
            // All languages support basic types
            (_, DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64) => true,
            (_, DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64) => true,
            (_, DataType::Float32 | DataType::Float64 | DataType::Bool) => true,

            // Vector types supported by most languages
            (Language::Rust | Language::Zig | Language::Julia, DataType::Vec2f32 | DataType::Vec3f32 | DataType::Vec4f32) => true,
            (Language::Rust | Language::Zig | Language::Julia, DataType::Vec2f64 | DataType::Vec3f64 | DataType::Vec4f64) => true,

            // Arrays supported by all languages
            (_, DataType::Array(_, _)) => true,

            // Structs supported by most languages (Elixir uses maps)
            (Language::Elixir, DataType::Struct(_)) => true, // Convert to map
            (_, DataType::Struct(_)) => true,

            // Language-specific types
            (Language::Rust, DataType::RustSpecific(_)) => true,
            (Language::Elixir, DataType::ElixirSpecific(_)) => true,
            (Language::Julia, DataType::JuliaSpecific(_)) => true,
            (Language::Zig, DataType::ZigSpecific(_)) => true,
            (Language::Nim, DataType::NimSpecific(_)) => true,

            _ => false,
        }
    }

    fn check_specific_conversion_issues(&self,
        source_lang: Language,
        target_lang: Language,
        data_type: &DataType
    ) -> Result<(), BindingError> {
        // Julia uses column-major arrays, others use row-major
        if matches!(data_type, DataType::Array(_, _)) {
            match (source_lang, target_lang) {
                (Language::Julia, lang) | (lang, Language::Julia)
                    if lang != Language::Julia => {
                    // This is a warning rather than an error - data will be transposed
                    log::warn!("Array layout conversion required between Julia and {:?}", lang);
                },
                _ => {}
            }
        }

        // Elixir doesn't have native struct types, uses maps
        if matches!(data_type, DataType::Struct(_)) {
            match (source_lang, target_lang) {
                (Language::Elixir, _) | (_, Language::Elixir) => {
                    log::info!("Struct will be converted to/from Elixir map");
                },
                _ => {}
            }
        }

        Ok(())
    }
}

// Re-export language-specific modules
pub use rust::RustBinding;
pub use elixir::ElixirBinding;
pub use julia::JuliaBinding;
pub use zig::ZigBinding;
pub use nim::NimBinding;