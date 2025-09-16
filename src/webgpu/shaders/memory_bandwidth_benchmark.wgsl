// Memory Bandwidth Benchmark Shader for WebGPU Performance Testing
// Tests memory-intensive workloads for memory bandwidth profiling

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct Uniforms {
    data_size: u32,
    stride: u32,
    iterations: u32,
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let size = uniforms.data_size;
    let stride = uniforms.stride;

    if (index >= size) {
        return;
    }

    var accumulator: f32 = 0.0;

    // Memory-intensive operations with strided access patterns
    for (var iter: u32 = 0u; iter < uniforms.iterations; iter++) {
        // Sequential reads
        for (var i: u32 = 0u; i < 32u; i++) {
            let read_index = (index + i) % size;
            accumulator += input_data[read_index];
        }

        // Strided reads to test memory bandwidth under different access patterns
        for (var i: u32 = 0u; i < 32u; i++) {
            let strided_index = (index + (i * stride)) % size;
            accumulator += input_data[strided_index] * 0.5;
        }

        // Random-like access pattern
        for (var i: u32 = 0u; i < 16u; i++) {
            let random_index = ((index * 31u + i * 17u) ^ (iter * 13u)) % size;
            accumulator += input_data[random_index] * 0.25;
        }
    }

    // Write result (tests write bandwidth)
    output_data[index] = accumulator;

    // Additional writes to stress memory bandwidth
    if (index + 1u < size) {
        output_data[index + 1u] = accumulator * 0.1;
    }
    if (index + 2u < size) {
        output_data[index + 2u] = accumulator * 0.01;
    }
}