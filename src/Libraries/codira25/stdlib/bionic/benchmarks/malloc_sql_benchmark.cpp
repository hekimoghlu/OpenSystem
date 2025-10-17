/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#include <malloc.h>
#include <stdlib.h>
#include <unistd.h>

#include <benchmark/benchmark.h>
#include "ScopedDecayTimeRestorer.h"
#include "util.h"

#if defined(__BIONIC__)

enum AllocEnum : uint8_t {
  MALLOC = 0,
  CALLOC,
  MEMALIGN,
  REALLOC,
  FREE,
};

struct MallocEntry {
  AllocEnum type;
  size_t idx;
  size_t size;
  size_t arg2;
};

void BenchmarkMalloc(MallocEntry entries[], size_t total_entries, size_t max_allocs) {
  void* ptrs[max_allocs];

  for (size_t i = 0; i < total_entries; i++) {
    switch (entries[i].type) {
    case MALLOC:
      ptrs[entries[i].idx] = malloc(entries[i].size);
      // Touch at least one byte of the allocation to make sure that
      // PSS for this allocation is counted.
      reinterpret_cast<uint8_t*>(ptrs[entries[i].idx])[0] = 10;
      break;
    case CALLOC:
      ptrs[entries[i].idx] = calloc(entries[i].arg2, entries[i].size);
      // Touch at least one byte of the allocation to make sure that
      // PSS for this allocation is counted.
      reinterpret_cast<uint8_t*>(ptrs[entries[i].idx])[0] = 20;
      break;
    case MEMALIGN:
      ptrs[entries[i].idx] = memalign(entries[i].arg2, entries[i].size);
      // Touch at least one byte of the allocation to make sure that
      // PSS for this allocation is counted.
      reinterpret_cast<uint8_t*>(ptrs[entries[i].idx])[0] = 30;
      break;
    case REALLOC:
      if (entries[i].arg2 == 0) {
        ptrs[entries[i].idx] = realloc(nullptr, entries[i].size);
      } else {
        ptrs[entries[i].idx] = realloc(ptrs[entries[i].arg2 - 1], entries[i].size);
      }
      // Touch at least one byte of the allocation to make sure that
      // PSS for this allocation is counted.
      reinterpret_cast<uint8_t*>(ptrs[entries[i].idx])[0] = 40;
      break;
    case FREE:
      free(ptrs[entries[i].idx]);
      break;
    }
  }
}

// This codifies playing back a single threaded trace of the allocations
// when running the SQLite BenchMark app.
// Instructions for recreating:
//   - Enable malloc debug
//       setprop wrap.com.wtsang02.sqliteutil "LIBC_DEBUG_MALLOC_OPTIONS=record_allocs logwrapper"
//   - Start the SQLite BenchMark app
//   - Dump allocs using the signal to get rid of non sql allocs(kill -47 <SQLITE_PID>)
//   - Run the benchmark.
//   - Dump allocs using the signal again.
//   - Find the thread that has the most allocs and run the helper script
//       bionic/libc/malloc_debug/tools/gen_malloc.pl -i <THREAD_ID> g_sql_entries kMaxSqlAllocSlots < <ALLOC_FILE> > malloc_sql.h
#include "malloc_sql.h"

static void BM_malloc_sql_trace_default(benchmark::State& state) {
  ScopedDecayTimeRestorer restorer;

  // The default is expected to be a zero decay time.
  mallopt(M_DECAY_TIME, 0);

  for (auto _ : state) {
    BenchmarkMalloc(g_sql_entries, sizeof(g_sql_entries) / sizeof(MallocEntry),
                    kMaxSqlAllocSlots);
  }
}
BIONIC_BENCHMARK(BM_malloc_sql_trace_default);

static void BM_malloc_sql_trace_decay1(benchmark::State& state) {
  ScopedDecayTimeRestorer restorer;

  mallopt(M_DECAY_TIME, 1);

  for (auto _ : state) {
    BenchmarkMalloc(g_sql_entries, sizeof(g_sql_entries) / sizeof(MallocEntry),
                    kMaxSqlAllocSlots);
  }
}
BIONIC_BENCHMARK(BM_malloc_sql_trace_decay1);

#endif
