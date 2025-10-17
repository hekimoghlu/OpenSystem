/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include <stdlib.h>

#include "spawn_benchmark.h"

#if defined(__LP64__)
static constexpr const char* kNativeTestDir = "nativetest64";
#else
static constexpr const char* kNativeTestDir = "nativetest";
#endif

static void BM_linker_relocation(benchmark::State& state) {
  std::string main = test_program("linker_reloc_bench_main");

  // Translate from:
  //    /data/benchmarktest[64]/linker-reloc-bench    [exe dir]
  // to:
  //    /data/nativetest[64]/linker-reloc-bench       [dir with test libs]
  std::string test_lib_dir = 
      android::base::Dirname(android::base::Dirname(android::base::GetExecutableDirectory())) +
      "/" + kNativeTestDir + "/linker-reloc-bench";

  setenv("LD_LIBRARY_PATH", test_lib_dir.c_str(), 1);

  BM_spawn_test(state, (const char*[]) { main.c_str(), nullptr });
}

BENCHMARK(BM_linker_relocation)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
