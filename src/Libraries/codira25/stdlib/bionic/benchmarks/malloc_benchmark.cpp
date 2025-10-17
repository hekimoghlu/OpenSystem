/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
#include <unistd.h>

#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>
#include "ScopedDecayTimeRestorer.h"
#include "util.h"

#if defined(__BIONIC__)

static void RunMalloptPurge(benchmark::State& state, int purge_value) {
  ScopedDecayTimeRestorer restorer;

  static size_t sizes[] = {8, 16, 32, 64, 128, 1024, 4096, 16384, 65536, 131072, 1048576};
  static int pagesize = getpagesize();
  mallopt(M_DECAY_TIME, 1);
  mallopt(M_PURGE_ALL, 0);
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<void*> ptrs;
    for (auto size : sizes) {
      // Allocate at least two pages worth of the allocations.
      for (size_t allocated = 0; allocated < 2 * static_cast<size_t>(pagesize); allocated += size) {
        void* ptr = malloc(size);
        if (ptr == nullptr) {
          state.SkipWithError("Failed to allocate memory");
        }
        MakeAllocationResident(ptr, size, pagesize);
        ptrs.push_back(ptr);
      }
    }
    // Free the memory, which should leave many of the pages resident until
    // the purge call.
    for (auto ptr : ptrs) {
      free(ptr);
    }
    ptrs.clear();
    state.ResumeTiming();

    mallopt(purge_value, 0);
  }
}

static void RunThreadsThroughput(benchmark::State& state, size_t size, size_t num_threads) {
  constexpr size_t kMaxBytes = 1 << 24;
  constexpr size_t kMaxThreads = 8;
  constexpr size_t kMinRounds = 4;
  const size_t MaxAllocCounts = kMaxBytes / size;
  std::mutex m;
  bool ready = false;
  std::condition_variable cv;
  std::thread* threads[kMaxThreads];

  // The goal is to create malloc/free interleaving patterns across threads.
  // The bytes processed by each thread will be the same. The difference is the
  // patterns. Here's an example:
  //
  // A: Allocation
  // D: Deallocation
  //
  //   T1    T2    T3
  //   A     A     A
  //   A     A     D
  //   A     D     A
  //   A     D     D
  //   D     A     A
  //   D     A     D
  //   D     D     A
  //   D     D     D
  //
  // To do this, `AllocCounts` and `AllocRounds` will be adjusted according to the
  // thread id.
  auto thread_task = [&](size_t id) {
    {
      std::unique_lock lock(m);
      // Wait until all threads are created.
      cv.wait(lock, [&] { return ready; });
    }

    void** MemPool;
    const size_t AllocCounts = (MaxAllocCounts >> id);
    const size_t AllocRounds = (kMinRounds << id);
    MemPool = new void*[AllocCounts];

    for (size_t i = 0; i < AllocRounds; ++i) {
      for (size_t j = 0; j < AllocCounts; ++j) {
        void* ptr = malloc(size);
        MemPool[j] = ptr;
      }

      // Use a fix seed to reduce the noise of different round of benchmark.
      const unsigned seed = 33529;
      std::shuffle(MemPool, &MemPool[AllocCounts], std::default_random_engine(seed));

      for (size_t j = 0; j < AllocCounts; ++j) free(MemPool[j]);
    }

    delete[] MemPool;
  };

  for (auto _ : state) {
    state.PauseTiming();
    // Don't need to acquire the lock because no thread is created.
    ready = false;

    for (size_t i = 0; i < num_threads; ++i) threads[i] = new std::thread(thread_task, i);

    state.ResumeTiming();

    {
      std::unique_lock lock(m);
      ready = true;
    }

    cv.notify_all();

    for (size_t i = 0; i < num_threads; ++i) {
      threads[i]->join();
      delete threads[i];
    }
  }

  const size_t ThreadsBytesProcessed = kMaxBytes * kMinRounds * num_threads;
  state.SetBytesProcessed(ThreadsBytesProcessed * static_cast<size_t>(state.iterations()));
}

static void BM_mallopt_purge(benchmark::State& state) {
  RunMalloptPurge(state, M_PURGE);
}
BIONIC_BENCHMARK(BM_mallopt_purge);

static void BM_mallopt_purge_all(benchmark::State& state) {
  RunMalloptPurge(state, M_PURGE_ALL);
}
BIONIC_BENCHMARK(BM_mallopt_purge_all);

// Note that this will only test a single size class at a time so that we can
// observe the impact of contention more often.
#define BM_MALLOC_THREADS_THROUGHPUT(SIZE, NUM_THREADS)                                      \
  static void BM_malloc_threads_throughput_##SIZE##_##NUM_THREADS(benchmark::State& state) { \
    RunThreadsThroughput(state, SIZE, NUM_THREADS);                                          \
  }                                                                                          \
  BIONIC_BENCHMARK(BM_malloc_threads_throughput_##SIZE##_##NUM_THREADS);

// There are three block categories in Scudo, we choose 1 from each category.
BM_MALLOC_THREADS_THROUGHPUT(64, 2);
BM_MALLOC_THREADS_THROUGHPUT(64, 4);
BM_MALLOC_THREADS_THROUGHPUT(64, 8);
BM_MALLOC_THREADS_THROUGHPUT(512, 2);
BM_MALLOC_THREADS_THROUGHPUT(512, 4);
BM_MALLOC_THREADS_THROUGHPUT(512, 8);
BM_MALLOC_THREADS_THROUGHPUT(8192, 2);
BM_MALLOC_THREADS_THROUGHPUT(8192, 4);
BM_MALLOC_THREADS_THROUGHPUT(8192, 8);

#endif
