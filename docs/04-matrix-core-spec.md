# PRD-004: Matrix Core Specification

## Executive Summary

Matrix Cores provide specialized linear algebra acceleration using advanced Zig implementations, featuring zero-cost memory abstractions and real-time SIMD visualizations through Phoenix LiveView.

## Core Architecture

### Zig-Based Matrix Operations

```zig
// src/zig_memory/matrix_core.zig
const std = @import("std");
const hip = @import("hip.zig");
const builtin = @import("builtin");

pub const MatrixCoreConfig = struct {
    core_id: u32,
    simd_width: u32 = 64,
    tensor_units: u32 = 16,
    shared_memory_banks: u32 = 32,
    matrix_block_size: u32 = 128,
    precision_modes: []PrecisionMode,
    
    const PrecisionMode = enum {
        fp32,
        fp16, 
        bf16,
        int8,
        int4,
    };
};

pub const MatrixCore = struct {
    config: MatrixCoreConfig,
    device_context: *hip.DeviceContext,
    tensor_ops_cache: TensorOpsCache,
    simd_scheduler: SIMDScheduler,
    memory_pools: [4]MemoryPool,
    telemetry_collector: TelemetryCollector,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, config: MatrixCoreConfig) !Self {
        const device_context = try hip.DeviceContext.create(config.core_id);
        
        return Self{
            .config = config,
            .device_context = device_context,
            .tensor_ops_cache = try TensorOpsCache.init(allocator),
            .simd_scheduler = try SIMDScheduler.init(config.simd_width),
            .memory_pools = try initMemoryPools(allocator, config),
            .telemetry_collector = try TelemetryCollector.init(config.core_id),
        };
    }
    
    pub fn matmul(self: *Self, comptime T: type, a: []const T, b: []const T, c: []T, 
                  dims: MatrixDimensions) !void {
        const start_time = std.time.nanoTimestamp();
        
        // Determine optimal algorithm based on dimensions and precision
        const algorithm = comptime self.selectMatMulAlgorithm(T, dims);
        
        // Allocate GPU memory with optimal layout
        const gpu_a = try self.allocateMatrixBuffer(T, dims.m * dims.k, .row_major);
        const gpu_b = try self.allocateMatrixBuffer(T, dims.k * dims.n, .col_major);  
        const gpu_c = try self.allocateMatrixBuffer(T, dims.m * dims.n, .row_major);
        defer self.deallocateBuffer(gpu_a);
        defer self.deallocateBuffer(gpu_b);
        defer self.deallocateBuffer(gpu_c);
        
        // Copy data to GPU with memory coalescing
        try self.copyToGPUCoalesced(T, a, gpu_a);
        try self.copyToGPUCoalesced(T, b, gpu_b);
        
        // Launch matrix multiplication kernel
        const kernel_params = MatMulKernelParams{
            .algorithm = algorithm,
            .block_size = comptime calculateOptimalBlockSize(dims),
            .precision = T,
            .use_tensor_cores = comptime (T == f16 or T == hip.bfloat16),
        };
        
        try self.launchMatMulKernel(kernel_params, gpu_a, gpu_b, gpu_c, dims);
        
        // Copy result back to host
        try self.copyFromGPUCoalesced(T, gpu_c, c);
        
        // Record telemetry
        const execution_time = std.time.nanoTimestamp() - start_time;
        self.telemetry_collector.recordMatMul(dims, execution_time, algorithm);
    }
    
    fn selectMatMulAlgorithm(comptime T: type, dims: MatrixDimensions) MatMulAlgorithm {
        // Compile-time algorithm selection based on type and dimensions
        return switch (T) {
            f32 => if (dims.isLarge()) .cutlass_sgemm else .naive_sgemm,
            f16 => .tensor_core_hgemm,
            hip.bfloat16 => .tensor_core_bgemm,
            i8 => .int8_gemm,
            else => @compileError("Unsupported matrix type: " ++ @typeName(T)),
        };
    }
    
    fn launchMatMulKernel(self: *Self, params: MatMulKernelParams, 
                         a_ptr: hip.DevicePtr, b_ptr: hip.DevicePtr, c_ptr: hip.DevicePtr,
                         dims: MatrixDimensions) !void {
        // Get or compile optimized kernel
        const kernel = try self.tensor_ops_cache.getOrCompileKernel(params);
        
        // Calculate grid and block dimensions for optimal SIMD utilization
        const grid_dim = hip.Dim3{
            .x = @intCast((dims.n + params.block_size.x - 1) / params.block_size.x),
            .y = @intCast((dims.m + params.block_size.y - 1) / params.block_size.y),
            .z = 1,
        };
        
        const block_dim = params.block_size;
        
        // Calculate shared memory requirements
        const shared_mem_size = comptime calculateSharedMemorySize(params);
        
        // Launch kernel with telemetry hooks
        const stream = try self.createComputeStream();
        defer stream.destroy();
        
        try hip.launchKernel(
            kernel.function,
            grid_dim,
            block_dim,
            &[_]*const anyopaque{ &a_ptr, &b_ptr, &c_ptr, &dims },
            shared_mem_size,
            stream,
        );
        
        // Start asynchronous telemetry collection
        try self.startKernelTelemetryCollection(kernel.id, stream);
    }
    
    pub fn batchMatMul(self: *Self, comptime T: type, operations: []const BatchedMatMulOp(T)) !void {
        // Optimized batched matrix multiplication with SIMD scheduling
        
        // Group operations by similarity for better cache utilization
        const grouped_ops = try self.groupSimilarOperations(operations);
        
        for (grouped_ops) |group| {
            // Process similar operations together
            try self.processBatchGroup(T, group);
        }
    }
    
    fn processBatchGroup(self: *Self, comptime T: type, group: []const BatchedMatMulOp(T)) !void {
        // Allocate persistent GPU memory for the batch
        const batch_memory = try self.allocateBatchMemory(T, group);
        defer self.deallocateBatchMemory(batch_memory);
        
        // Schedule SIMD operations across tensor units
        const schedule = try self.simd_scheduler.scheduleBatch(group);
        
        for (schedule.waves) |wave| {
            // Launch wave of operations simultaneously
            var streams: [16]hip.Stream = undefined;
            
            for (wave.operations, 0..) |op, i| {
                streams[i] = try self.createComputeStream();
                try self.launchMatMulKernelAsync(op, streams[i]);
            }
            
            // Wait for wave completion
            for (wave.operations, 0..) |_, i| {
                try streams[i].synchronize();
                streams[i].destroy();
            }
        }
    }
    
    pub fn getTelemetry(self: *Self) MatrixCoreTelemetry {
        return self.telemetry_collector.collectCurrentTelemetry();
    }
    
    pub fn getDetailedTelemetry(self: *Self) DetailedMatrixTelemetry {
        return DetailedMatrixTelemetry{
            .core_id = self.config.core_id,
            .timestamp = std.time.nanoTimestamp(),
            
            // SIMD utilization per lane
            .simd_utilization = self.simd_scheduler.getSIMDUtilization(),
            
            // Tensor core utilization
            .tensor_core_utilization = self.getTensorCoreUtilization(),
            
            // Active matrix operations
            .active_operations = self.getActiveOperations(),
            
            // Memory bandwidth utilization
            .memory_bandwidth_utilization = self.getMemoryBandwidthUtilization(),
            
            // Cache statistics
            .l1_cache_hit_rate = self.getL1CacheHitRate(),
            .l2_cache_hit_rate = self.getL2CacheHitRate(),
            
            // Operation throughput
            .gemm_throughput = self.calculateGEMMThroughput(), // TOPS
            .memory_throughput = self.calculateMemoryThroughput(), // GB/s
            
            // Precision mode statistics
            .precision_distribution = self.getPrecisionDistribution(),
            
            // Performance bottlenecks
            .bottleneck_analysis = self.analyzeBottlenecks(),
            
            // Optimization opportunities
            .optimization_hints = self.generateOptimizationHints(),
        };
    }
};

// Zero-cost compile-time matrix operation optimization
pub fn OptimizedMatMul(comptime T: type, comptime M: u32, comptime K: u32, comptime N: u32) type {
    return struct {
        const Self = @This();
        
        // Compile-time constants for optimization
        const BLOCK_SIZE_M = comptime calculateOptimalBlockSizeM(M, K, N);
        const BLOCK_SIZE_K = comptime calculateOptimalBlockSizeK(M, K, N);
        const BLOCK_SIZE_N = comptime calculateOptimalBlockSizeN(M, K, N);
        const USE_TENSOR_CORES = comptime (T == f16 or T == hip.bfloat16);
        const VECTORIZATION_WIDTH = comptime calculateVectorizationWidth(T);
        
        pub fn execute(a: [M * K]T, b: [K * N]T) [M * N]T {
            var c: [M * N]T = undefined;
            
            // Unrolled loops for small matrices
            if (comptime M * K * N < 1024) {
                comptime var i = 0;
                inline while (i < M) : (i += 1) {
                    comptime var j = 0;
                    inline while (j < N) : (j += 1) {
                        var sum: T = 0;
                        comptime var k = 0;
                        inline while (k < K) : (k += 1) {
                            sum += a[i * K + k] * b[k * N + j];
                        }
                        c[i * N + j] = sum;
                    }
                }
            } else {
                // Blocked algorithm for larger matrices
                executeBlockedMatMul(&a, &b, &c);
            }
            
            return c;
        }
        
        fn executeBlockedMatMul(a: *const [M * K]T, b: *const [K * N]T, c: *[M * N]T) void {
            // Tiled matrix multiplication with compile-time optimization
            comptime var ii = 0;
            inline while (ii < M) : (ii += BLOCK_SIZE_M) {
                comptime var jj = 0;
                inline while (jj < N) : (jj += BLOCK_SIZE_N) {
                    comptime var kk = 0;
                    inline while (kk < K) : (kk += BLOCK_SIZE_K) {
                        // Process tile
                        processTile(a, b, c, ii, jj, kk);
                    }
                }
            }
        }
        
        fn processTile(a: *const [M * K]T, b: *const [K * N]T, c: *[M * N]T, 
                      ii: u32, jj: u32, kk: u32) void {
            const max_i = @min(ii + BLOCK_SIZE_M, M);
            const max_j = @min(jj + BLOCK_SIZE_N, N);
            const max_k = @min(kk + BLOCK_SIZE_K, K);
            
            var i = ii;
            while (i < max_i) : (i += 1) {
                var j = jj;
                while (j < max_j) : (j += VECTORIZATION_WIDTH) {
                    // Vectorized inner loop
                    var sum_vec: @Vector(VECTORIZATION_WIDTH, T) = @splat(0);
                    
                    var k = kk;
                    while (k < max_k) : (k += 1) {
                        const a_elem = a[i * K + k];
                        const b_vec: @Vector(VECTORIZATION_WIDTH, T) = b[k * N + j..k * N + j + VECTORIZATION_WIDTH][0..VECTORIZATION_WIDTH].*;
                        sum_vec += @splat(a_elem) * b_vec;
                    }
                    
                    // Store result
                    const c_slice = c[i * N + j..i * N + j + VECTORIZATION_WIDTH][0..VECTORIZATION_WIDTH];
                    c_slice.* += sum_vec;
                }
            }
        }
    };
}

// SIMD Utilization Scheduler
const SIMDScheduler = struct {
    simd_width: u32,
    utilization_map: []f32,
    active_operations: std.ArrayList(SIMDOperation),
    
    const SIMDOperation = struct {
        id: u64,
        lanes_required: u32,
        execution_time: u64,
        priority: u8,
    };
    
    pub fn init(simd_width: u32) !SIMDScheduler {
        return SIMDScheduler{
            .simd_width = simd_width,
            .utilization_map = try std.heap.page_allocator.alloc(f32, simd_width),
            .active_operations = std.ArrayList(SIMDOperation).init(std.heap.page_allocator),
        };
    }
    
    pub fn scheduleBatch(self: *SIMDScheduler, operations: []const BatchedMatMulOp) !BatchSchedule {
        // Advanced SIMD lane allocation algorithm
        var schedule = BatchSchedule.init();
        
        // Sort operations by resource requirements
        const sorted_ops = try self.sortByResourceRequirements(operations);
        
        // Pack operations into SIMD waves
        for (sorted_ops) |op| {
            const available_lanes = self.findAvailableLanes(op.simd_requirements);
            
            if (available_lanes) |lanes| {
                try schedule.addToCurrentWave(op, lanes);
                self.markLanesUsed(lanes);
            } else {
                // Start new wave
                try schedule.startNewWave();
                self.resetLaneUtilization();
                try schedule.addToCurrentWave(op, self.allocateLanes(op.simd_requirements));
            }
        }
        
        return schedule;
    }
    
    pub fn getSIMDUtilization(self: *SIMDScheduler) []const f32 {
        return self.utilization_map;
    }
};

// Tensor Operations Cache
const TensorOpsCache = struct {
    compiled_kernels: std.HashMap(KernelParams, CompiledKernel, KernelParamsContext, 80),
    
    const KernelParams = struct {
        algorithm: MatMulAlgorithm,
        precision: type,
        block_size: hip.Dim3,
        use_tensor_cores: bool,
    };
    
    pub fn getOrCompileKernel(self: *TensorOpsCache, params: MatMulKernelParams) !*CompiledKernel {
        const key = KernelParams{
            .algorithm = params.algorithm,
            .precision = params.precision,
            .block_size = params.block_size,
            .use_tensor_cores = params.use_tensor_cores,
        };
        
        if (self.compiled_kernels.get(key)) |kernel| {
            return kernel;
        }
        
        // Compile new kernel with Zig metaprogramming
        const kernel_source = comptime generateKernelSource(params);
        const compiled_kernel = try self.compileKernel(kernel_source, params);
        
        try self.compiled_kernels.put(key, compiled_kernel);
        return self.compiled_kernels.getPtr(key).?;
    }
};

// Compile-time kernel generation
fn generateKernelSource(comptime params: MatMulKernelParams) []const u8 {
    return switch (params.algorithm) {
        .tensor_core_hgemm => generateTensorCoreKernel(f16, params),
        .tensor_core_bgemm => generateTensorCoreKernel(hip.bfloat16, params),
        .cutlass_sgemm => generateCutlassKernel(f32, params),
        .naive_sgemm => generateNaiveKernel(f32, params),
        .int8_gemm => generateInt8Kernel(params),
    };
}

fn generateTensorCoreKernel(comptime T: type, comptime params: MatMulKernelParams) []const u8 {
    return 
        \\#include <hip/hip_runtime.h>
        \\#include <hip/hip_fp16.h>
        \\
        \\__global__ void tensor_core_matmul(
        \\    const {} *a, const {} *b, {} *c,
        \\    int m, int n, int k
        \\) {{
        \\    // Tensor core implementation with {} precision
        \\    extern __shared__ {} smem[];
        \\    
        \\    // Thread mapping optimized for tensor cores
        \\    const int warp_id = threadIdx.x / 32;
        \\    const int lane_id = threadIdx.x % 32;
        \\    
        \\    // Use mma instructions for optimal throughput
        \\    #pragma unroll
        \\    for (int tile_k = 0; tile_k < k; tile_k += {}) {{
        \\        // Load tiles into shared memory
        \\        // Perform tensor core mma operations
        \\        // Accumulate results
        \\    }}
        \\    
        \\    // Store results with memory coalescing
        \\}}
    ;
    // Template would be filled with actual type names and parameters
}
```

### Real-Time SIMD Visualization

```elixir
# lib/amdgpu/nif/matrix_core.ex
defmodule AMDGPU.NIF.MatrixCore do
  @moduledoc """
  Elixir NIF interface for Matrix Core functionality with real-time SIMD monitoring
  """
  
  @on_load :load_nifs
  
  def load_nifs do
    :erlang.load_nif('./priv/matrix_core', 0)
  end
  
  # Core operations
  def initialize_core(_config), do: :erlang.nif_error(:nif_not_loaded)
  def matmul(_core_id, _a, _b, _dims), do: :erlang.nif_error(:nif_not_loaded)
  def batch_matmul(_core_id, _operations), do: :erlang.nif_error(:nif_not_loaded)
  
  # SIMD telemetry
  def get_simd_utilization(_core_id), do: :erlang.nif_error(:nif_not_loaded)
  def get_detailed_telemetry(_core_id), do: :erlang.nif_error(:nif_not_loaded)
  def subscribe_to_simd_telemetry(_core_id, _callback_pid), do: :erlang.nif_error(:nif_not_loaded)
  
  # High-level matrix operations API
  def gemm(core_id, a, b, options \\ []) do
    dims = %{
      m: options[:m] || length(a),
      k: options[:k] || length(Enum.at(a, 0, [])), 
      n: options[:n] || length(Enum.at(b, 0, []))
    }
    
    precision = options[:precision] || :fp32
    algorithm = determine_optimal_algorithm(dims, precision)
    
    matmul(core_id, a, b, Map.put(dims, :algorithm, algorithm))
  end
  
  def batched_gemm(core_id, matrices, options \\ []) do
    # Optimize batch processing
    optimized_operations = optimize_batch_operations(matrices, options)
    batch_matmul(core_id, optimized_operations)
  end
  
  defp determine_optimal_algorithm(dims, precision) do
    cond do
      precision in [:fp16, :bf16] and tensor_cores_available?() ->
        :tensor_core
        
      dims.m * dims.k * dims.n > 1_000_000 ->
        :cutlass
        
      true ->
        :naive
    end
  end
  
  defp optimize_batch_operations(matrices, options) do
    # Group similar matrices for better cache utilization
    matrices
    |> Enum.group_by(&get_matrix_signature/1)
    |> Enum.flat_map(fn {_signature, group} ->
      optimize_matrix_group(group, options)
    end)
  end
  
  defp tensor_cores_available? do
    # Check hardware capability
    true # Assume available for now
  end
end
```

### Phoenix LiveView Matrix Dashboard

```elixir
# lib/amdgpu_web/live/matrix_core_live.ex
defmodule AMDGPUWeb.MatrixCoreLive do
  use AMDGPUWeb, :live_view
  alias AMDGPU.NIF.MatrixCore
  
  @impl true
  def mount(%{"core_id" => core_id}, _session, socket) do
    core_id = String.to_integer(core_id)
    
    if connected?(socket) do
      # Subscribe to SIMD telemetry
      AMDGPUWeb.Endpoint.subscribe("matrix_core:#{core_id}")
      MatrixCore.subscribe_to_simd_telemetry(core_id, self())
      
      # High-frequency SIMD updates
      :timer.send_interval(16, self(), :update_simd_visualization)
    end
    
    initial_state = %{
      core_id: core_id,
      simd_utilization: List.duplicate(0.0, 64), # 64 SIMD lanes
      tensor_core_utilization: List.duplicate(0.0, 16), # 16 tensor cores
      active_operations: [],
      throughput_history: [],
      precision_distribution: %{fp32: 0, fp16: 0, bf16: 0, int8: 0},
      memory_bandwidth: 0.0,
      cache_statistics: %{l1_hit_rate: 0.0, l2_hit_rate: 0.0},
      bottleneck_analysis: [],
      last_update: System.system_time(:millisecond)
    }
    
    {:ok, assign(socket, initial_state)}
  end
  
  @impl true
  def handle_info(:update_simd_visualization, socket) do
    case MatrixCore.get_detailed_telemetry(socket.assigns.core_id) do
      {:ok, telemetry} ->
        socket = socket
        |> assign(:simd_utilization, telemetry.simd_utilization)
        |> assign(:tensor_core_utilization, telemetry.tensor_core_utilization)
        |> assign(:active_operations, telemetry.active_operations)
        |> assign(:precision_distribution, telemetry.precision_distribution)
        |> assign(:memory_bandwidth, telemetry.memory_throughput)
        |> assign(:cache_statistics, %{
            l1_hit_rate: telemetry.l1_cache_hit_rate,
            l2_hit_rate: telemetry.l2_cache_hit_rate
          })
        |> assign(:bottleneck_analysis, telemetry.bottleneck_analysis)
        |> update(:throughput_history, fn history ->
            new_point = %{
              timestamp: System.system_time(:millisecond),
              gemm_throughput: telemetry.gemm_throughput
            }
            [new_point | Enum.take(history, 299)] # Keep 5 seconds at 60 FPS
          end)
        
        {:noreply, socket}
        
      {:error, _reason} ->
        {:noreply, socket}
    end
  end
  
  @impl true 
  def handle_event("launch_benchmark", %{"matrix_size" => size_str}, socket) do
    size = String.to_integer(size_str)
    
    # Generate test matrices
    a = generate_test_matrix(size, size)
    b = generate_test_matrix(size, size)
    
    # Launch matrix multiplication benchmark
    Task.start(fn ->
      start_time = System.monotonic_time(:microsecond)
      
      case MatrixCore.gemm(socket.assigns.core_id, a, b, precision: :fp16) do
        {:ok, _result} ->
          end_time = System.monotonic_time(:microsecond)
          execution_time = end_time - start_time
          
          # Broadcast benchmark result
          AMDGPUWeb.Endpoint.broadcast("matrix_core:#{socket.assigns.core_id}", 
            "benchmark_complete", %{
              matrix_size: size,
              execution_time: execution_time,
              throughput: calculate_throughput(size, execution_time)
            })
            
        {:error, reason} ->
          AMDGPUWeb.Endpoint.broadcast("matrix_core:#{socket.assigns.core_id}",
            "benchmark_error", %{reason: reason})
      end
    end)
    
    {:noreply, put_flash(socket, :info, "Benchmark started for #{size}x#{size} matrices")}
  end
  
  @impl true
  def render(assigns) do
    ~H"""
    <div class="matrix-core-monitor" data-core-id={@core_id}>
      <div class="core-header">
        <h2>Matrix Core <%= @core_id %></h2>
        <div class="core-metrics">
          <span>Memory BW: <%= Float.round(@memory_bandwidth, 1) %> GB/s</span>
          <span>Active Ops: <%= length(@active_operations) %></span>
        </div>
      </div>
      
      <!-- SIMD Lane Utilization Heatmap -->
      <div class="simd-visualization">
        <h3>SIMD Lane Utilization (64 lanes)</h3>
        <div class="simd-heatmap" phx-hook="SIMDHeatmap" 
             data-utilization={Jason.encode!(@simd_utilization)}>
        </div>
        <div class="simd-legend">
          <span class="legend-item low">0%</span>
          <span class="legend-item medium">50%</span> 
          <span class="legend-item high">100%</span>
        </div>
      </div>
      
      <!-- Tensor Core Utilization -->
      <div class="tensor-cores">
        <h3>Tensor Core Utilization</h3>
        <div class="tensor-core-grid">
          <%= for {utilization, idx} <- Enum.with_index(@tensor_core_utilization) do %>
            <div class="tensor-core" data-core-id={idx} 
                 data-utilization={utilization}
                 style={"background: hsl(#{utilization * 120}, 100%, 50%)"}>
              <span class="core-id">TC<%= idx %></span>
              <span class="utilization"><%= Float.round(utilization * 100, 1) %>%</span>
            </div>
          <% end %>
        </div>
      </div>
      
      <!-- Active Matrix Operations -->
      <div class="active-operations">
        <h3>Active Matrix Operations</h3>
        <%= if length(@active_operations) > 0 do %>
          <div class="operations-list">
            <%= for op <- @active_operations do %>
              <div class="operation-item" data-op-id={op.id}>
                <div class="operation-header">
                  <span class="operation-type"><%= op.type %></span>
                  <span class="dimensions"><%= op.dimensions %></span>
                  <span class="precision"><%= op.precision %></span>
                </div>
                <div class="operation-progress">
                  <div class="progress-bar" style={"width: #{op.progress}%"}></div>
                  <span class="progress-text"><%= Float.round(op.progress, 1) %>%</span>
                </div>
                <div class="operation-stats">
                  <span>Throughput: <%= Float.round(op.throughput, 2) %> TOPS</span>
                  <span>SIMD Lanes: <%= op.simd_lanes_used %>/64</span>
                  <span>Tensor Cores: <%= op.tensor_cores_used %>/16</span>
                </div>
              </div>
            <% end %>
          </div>
        <% else %>
          <div class="no-operations">No active matrix operations</div>
        <% end %>
      </div>
      
      <!-- Throughput History Chart -->
      <div class="throughput-chart">
        <h3>GEMM Throughput History</h3>
        <canvas id="throughput-chart" phx-hook="ThroughputChart"
                data-history={Jason.encode!(@throughput_history)}>
        </canvas>
      </div>
      
      <!-- Precision Mode Distribution -->
      <div class="precision-distribution">
        <h3>Precision Mode Distribution</h3>
        <div class="precision-bars">
          <%= for {precision, count} <- @precision_distribution do %>
            <div class="precision-bar">
              <span class="precision-label"><%= precision %></span>
              <div class="bar-container">
                <div class="bar" style={"width: #{count}%"}></div>
              </div>
              <span class="precision-count"><%= count %>%</span>
            </div>
          <% end %>
        </div>
      </div>
      
      <!-- Cache Statistics -->
      <div class="cache-stats">
        <h3>Cache Performance</h3>
        <div class="cache-grid">
          <div class="cache-stat">
            <span class="label">L1 Hit Rate:</span>
            <span class="value"><%= Float.round(@cache_statistics.l1_hit_rate * 100, 1) %>%</span>
          </div>
          <div class="cache-stat">
            <span class="label">L2 Hit Rate:</span>
            <span class="value"><%= Float.round(@cache_statistics.l2_hit_rate * 100, 1) %>%</span>
          </div>
        </div>
      </div>
      
      <!-- Performance Bottlenecks -->
      <%= if length(@bottleneck_analysis) > 0 do %>
        <div class="bottleneck-analysis">
          <h3>Performance Bottlenecks</h3>
          <%= for bottleneck <- @bottleneck_analysis do %>
            <div class="bottleneck-item" data-severity={bottleneck.severity}>
              <div class="bottleneck-type"><%= bottleneck.type %></div>
              <div class="bottleneck-description"><%= bottleneck.description %></div>
              <div class="bottleneck-impact">Impact: <%= bottleneck.performance_impact %>%</div>
            </div>
          <% end %>
        </div>
      <% end %>
      
      <!-- Benchmark Controls -->
      <div class="benchmark-controls">
        <h3>Matrix Multiplication Benchmarks</h3>
        <div class="benchmark-buttons">
          <button type="button" phx-click="launch_benchmark" 
                  phx-value-matrix_size="512" class="btn-primary">
            512x512
          </button>
          <button type="button" phx-click="launch_benchmark"
                  phx-value-matrix_size="1024" class="btn-primary">
            1024x1024  
          </button>
          <button type="button" phx-click="launch_benchmark"
                  phx-value-matrix_size="2048" class="btn-primary">
            2048x2048
          </button>
          <button type="button" phx-click="launch_benchmark"
                  phx-value-matrix_size="4096" class="btn-primary">
            4096x4096
          </button>
        </div>
      </div>
    </div>
    """
  end
  
  defp generate_test_matrix(rows, cols) do
    # Generate random test matrix
    for _i <- 1..rows do
      for _j <- 1..cols do
        :rand.uniform() * 2.0 - 1.0 # Random values between -1 and 1
      end
    end
  end
  
  defp calculate_throughput(matrix_size, execution_time_microseconds) do
    # Calculate TOPS (Tera Operations Per Second)
    operations = 2 * matrix_size * matrix_size * matrix_size # 2 ops per multiply-add
    execution_time_seconds = execution_time_microseconds / 1_000_000
    operations / execution_time_seconds / 1_000_000_000_000 # Convert to TOPS
  end
end
```

### Advanced JavaScript Hooks for Matrix Visualization

```javascript
// assets/js/matrix_visualization_hooks.js

export const SIMDHeatmap = {
  mounted() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 512; // 8x64 SIMD lanes
    this.canvas.height = 64;
    this.el.appendChild(this.canvas);
    
    this.ctx = this.canvas.getContext('2d');
    this.updateHeatmap();
  },
  
  updated() {
    this.updateHeatmap();
  },
  
  updateHeatmap() {
    const utilizationData = JSON.parse(this.el.dataset.utilization);
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    // Create ImageData for efficient pixel manipulation
    const imageData = this.ctx.createImageData(width, height);
    const data = imageData.data;
    
    for (let lane = 0; lane < 64; lane++) {
      const utilization = utilizationData[lane] || 0;
      
      // Calculate position in 8x8 grid per lane
      const laneX = (lane % 8) * 64; // Each lane gets 64 pixels wide
      const laneY = Math.floor(lane / 8) * 8; // Each lane gets 8 pixels tall
      
      // Color based on utilization (green = high, red = low)
      const red = Math.floor((1 - utilization) * 255);
      const green = Math.floor(utilization * 255);
      const blue = 0;
      
      // Fill the lane's pixels
      for (let y = laneY; y < laneY + 8; y++) {
        for (let x = laneX; x < laneX + 64; x++) {
          const pixelIndex = (y * width + x) * 4;
          data[pixelIndex] = red;     // R
          data[pixelIndex + 1] = green; // G  
          data[pixelIndex + 2] = blue;  // B
          data[pixelIndex + 3] = 255;   // A
        }
      }
    }
    
    this.ctx.putImageData(imageData, 0, 0);
    
    // Add lane labels
    this.ctx.fillStyle = 'white';
    this.ctx.font = '10px monospace';
    for (let lane = 0; lane < 64; lane++) {
      const laneX = (lane % 8) * 64 + 2;
      const laneY = Math.floor(lane / 8) * 8 + 12;
      this.ctx.fillText(`${lane}`, laneX, laneY);
    }
  }
};

export const ThroughputChart = {
  mounted() {
    this.initChart();
  },
  
  updated() {
    this.updateChart();
  },
  
  initChart() {
    const ctx = this.el.getContext('2d');
    
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'GEMM Throughput (TOPS)',
          data: [],
          borderColor: '#00ff41',
          backgroundColor: 'rgba(0, 255, 65, 0.1)',
          tension: 0.4,
          pointRadius: 0, // Hide points for smooth line
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { 
            type: 'linear',
            position: 'bottom',
            title: { display: true, text: 'Time (ms)' }
          },
          y: { 
            min: 0,
            title: { display: true, text: 'Throughput (TOPS)' }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  },
  
  updateChart() {
    const history = JSON.parse(this.el.dataset.history);
    
    // Convert to Chart.js format
    const chartData = history.map(point => ({
      x: point.timestamp,
      y: point.gemm_throughput
    }));
    
    this.chart.data.datasets[0].data = chartData;
    this.chart.update('none'); // No animation for real-time updates
  }
};

export const MatrixOperationViz = {
  mounted() {
    this.initVisualization();
  },
  
  initVisualization() {
    // Three.js 3D visualization of matrix operations
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, this.el.clientWidth / this.el.clientHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.el.clientWidth, this.el.clientHeight);
    this.el.appendChild(this.renderer.domElement);
    
    // Create matrix representations
    this.matrixA = this.createMatrixMesh(0xff4444); // Red
    this.matrixB = this.createMatrixMesh(0x4444ff); // Blue  
    this.matrixC = this.createMatrixMesh(0x44ff44); // Green
    
    this.matrixA.position.set(-2, 0, 0);
    this.matrixB.position.set(0, 0, 0);
    this.matrixC.position.set(2, 0, 0);
    
    this.scene.add(this.matrixA);
    this.scene.add(this.matrixB);
    this.scene.add(this.matrixC);
    
    // Add lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1);
    this.scene.add(light);
    
    this.camera.position.z = 5;
    
    this.animate();
  },
  
  createMatrixMesh(color) {
    const geometry = new THREE.BoxGeometry(1, 1, 0.1);
    const material = new THREE.MeshLambertMaterial({ color });
    return new THREE.Mesh(geometry, material);
  },
  
  animate() {
    requestAnimationFrame(() => this.animate());
    
    // Rotate matrices to show activity
    this.matrixA.rotation.y += 0.01;
    this.matrixB.rotation.y += 0.01;
    this.matrixC.rotation.y += 0.01;
    
    this.renderer.render(this.scene, this.camera);
  }
};
```

## Key Matrix Core Features

1. **Zero-Cost Zig Abstractions**: Compile-time optimization with no runtime overhead
2. **Advanced SIMD Scheduling**: Optimal utilization of 64 SIMD lanes
3. **Tensor Core Integration**: Native support for mixed-precision operations
4. **Real-Time Heatmaps**: Live SIMD lane utilization visualization
5. **Batched Operations**: Optimized batch processing for throughput
6. **Cache-Aware Algorithms**: Memory access pattern optimization

## Performance Targets

- **Peak Throughput**: >100 TOPS (FP16 tensor operations)
- **SIMD Utilization**: >90% average lane utilization
- **Memory Efficiency**: >80% peak bandwidth utilization
- **Cache Hit Rates**: >85% L1, >70% L2
- **Batch Processing**: >10x speedup vs sequential operations
- **Monitoring Overhead**: <1% performance impact