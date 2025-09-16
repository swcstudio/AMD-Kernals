// Matrix Multiplication Benchmark Shader for WebGPU Performance Testing
// Tests compute-intensive workloads for adapter performance profiling

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_result: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

struct Uniforms {
    matrix_size: u32,
    iterations: u32,
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let size = uniforms.matrix_size;

    if (row >= size || col >= size) {
        return;
    }

    var result: f32 = 0.0;

    // Perform matrix multiplication with multiple iterations for benchmarking
    for (var iter: u32 = 0u; iter < uniforms.iterations; iter++) {
        for (var k: u32 = 0u; k < size; k++) {
            let a_index = row * size + k;
            let b_index = k * size + col;
            result += matrix_a[a_index] * matrix_b[b_index];
        }
    }

    let result_index = row * size + col;
    matrix_result[result_index] = result;
}