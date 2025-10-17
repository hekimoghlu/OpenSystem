/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#pragma once

#include <android-base/file.h>
#include <benchmark/benchmark.h>

void BM_spawn_test(benchmark::State& state, const char* const* argv);

static inline std::string test_program(const char* name) {
#if defined(__LP64__)
  return android::base::GetExecutableDirectory() + "/" + name + "64";
#else
  return android::base::GetExecutableDirectory() + "/" + name + "32";
#endif
}

#define SPAWN_BENCHMARK(name, ...)                                                    \
    BENCHMARK_CAPTURE(BM_spawn_test, name, (const char*[]) { __VA_ARGS__, nullptr })  \
        ->UseRealTime()                                                               \
        ->Unit(benchmark::kMicrosecond)                                               \
