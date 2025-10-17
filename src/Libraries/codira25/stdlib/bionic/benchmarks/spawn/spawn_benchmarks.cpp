/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#include "spawn_benchmark.h"

SPAWN_BENCHMARK(noop, test_program("bench_noop").c_str());
SPAWN_BENCHMARK(noop_nostl, test_program("bench_noop_nostl").c_str());
SPAWN_BENCHMARK(noop_static, test_program("bench_noop_static").c_str());
SPAWN_BENCHMARK(bench_cxa_atexit, test_program("bench_cxa_atexit").c_str(), "100000", "_Exit");
SPAWN_BENCHMARK(bench_cxa_atexit_full, test_program("bench_cxa_atexit").c_str(), "100000", "exit");

// Android has a /bin -> /system/bin symlink, but use /system/bin explicitly so we can more easily
// compare Bionic-vs-glibc on a Linux desktop machine.
#if defined(__GLIBC__)

SPAWN_BENCHMARK(bin_true, "/bin/true");
SPAWN_BENCHMARK(sh_true, "/bin/sh", "-c", "true");

#elif defined(__ANDROID__)

SPAWN_BENCHMARK(system_bin_true, "/system/bin/true");
SPAWN_BENCHMARK(vendor_bin_true, "/vendor/bin/true");
SPAWN_BENCHMARK(system_sh_true, "/system/bin/sh", "-c", "true");
SPAWN_BENCHMARK(vendor_sh_true, "/vendor/bin/sh", "-c", "true");

#endif

BENCHMARK_MAIN();
