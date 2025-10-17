/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <err.h>
#include <malloc.h>
#include <stdint.h>

#include <map>
#include <unordered_map>
#include <vector>

#include <benchmark/benchmark.h>
#include "util.h"

#include <android-base/strings.h>

#if defined(__BIONIC__)

#include <meminfo/procmeminfo.h>
#include <procinfo/process_map.h>

static void Gather(uint64_t* rss_bytes) {
  android::meminfo::ProcMemInfo proc_mem(getpid());
  const std::vector<android::meminfo::Vma>& maps = proc_mem.MapsWithoutUsageStats();
  for (auto& vma : maps) {
    if (vma.name == "[anon:libc_malloc]" || android::base::StartsWith(vma.name, "[anon:scudo:") ||
        android::base::StartsWith(vma.name, "[anon:GWP-ASan")) {
      android::meminfo::Vma update_vma(vma);
      if (!proc_mem.FillInVmaStats(update_vma)) {
        err(1, "FillInVmaStats failed");
      }
      *rss_bytes += update_vma.usage.rss;
    }
  }
}
#endif

template <typename MapType>
static void MapBenchmark(benchmark::State& state, size_t num_elements) {
#if defined(__BIONIC__)
  uint64_t rss_bytes = 0;
#endif

  for (auto _ : state) {
#if defined(__BIONIC__)
    state.PauseTiming();
    mallopt(M_PURGE_ALL, 0);
    uint64_t rss_bytes_before = 0;
    Gather(&rss_bytes_before);
    state.ResumeTiming();
#endif
    MapType map;
    for (size_t i = 0; i < num_elements; i++) {
      map[i][0] = 0;
    }
#if defined(__BIONIC__)
    state.PauseTiming();
    mallopt(M_PURGE_ALL, 0);
    Gather(&rss_bytes);
    // Try and record only the memory used in the map.
    rss_bytes -= rss_bytes_before;
    state.ResumeTiming();
#endif
  }

#if defined(__BIONIC__)
  double rss_mb = (rss_bytes / static_cast<double>(state.iterations())) / 1024.0 / 1024.0;
  state.counters["RSS_MB"] = rss_mb;
#endif
}

static void BM_std_map_8(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[8]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_map_8);

static void BM_std_map_16(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[16]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_map_16);

static void BM_std_map_32(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[32]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_map_32);

static void BM_std_map_64(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[64]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_map_64);

static void BM_std_map_96(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[96]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_map_96);

static void BM_std_map_128(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[128]>>(state, 500000);
}
BIONIC_BENCHMARK(BM_std_map_128);

static void BM_std_map_256(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[256]>>(state, 500000);
}
BIONIC_BENCHMARK(BM_std_map_256);

static void BM_std_map_512(benchmark::State& state) {
  MapBenchmark<std::map<uint64_t, char[512]>>(state, 500000);
}
BIONIC_BENCHMARK(BM_std_map_512);

static void BM_std_unordered_map_8(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[8]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_8);

static void BM_std_unordered_map_16(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[16]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_16);

static void BM_std_unordered_map_32(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[32]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_32);

static void BM_std_unordered_map_64(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[64]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_64);

static void BM_std_unordered_map_96(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[96]>>(state, 1000000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_96);

static void BM_std_unordered_map_128(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[128]>>(state, 500000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_128);

static void BM_std_unordered_map_256(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[256]>>(state, 500000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_256);

static void BM_std_unordered_map_512(benchmark::State& state) {
  MapBenchmark<std::unordered_map<uint64_t, char[512]>>(state, 500000);
}
BIONIC_BENCHMARK(BM_std_unordered_map_512);
