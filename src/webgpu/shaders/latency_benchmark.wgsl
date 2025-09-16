// Latency Benchmark Shader for WebGPU Performance Testing
// Tests minimal workload to measure execution latency overhead

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct Uniforms {
    data_size: u32,
    timestamp_start: u32,
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= uniforms.data_size) {
        return;
    }

    // Minimal computation to measure pure dispatch latency
    let value = input_data[index];

    // Simple operation that cannot be optimized away
    let result = value * 2.0 + 1.0;

    output_data[index] = result;

    // Ensure the computation is not eliminated by the optimizer
    if (result > 1000000.0) {
        output_data[index] = 0.0;
    }
}