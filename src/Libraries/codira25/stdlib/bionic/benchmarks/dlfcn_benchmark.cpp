/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include <android-base/strings.h>
#include <benchmark/benchmark.h>
#include <dlfcn.h>

#include "util.h"

void local_function() {}

template<typename F>
int bm_dladdr(F fun) {
    const void* addr = reinterpret_cast<void*>(fun);
    Dl_info info;
    int res = dladdr(addr, &info);
    if (res == 0) abort();
    if (info.dli_fname == nullptr) abort();

    // needed for DoNotOptimize
    return res;
}
BIONIC_TRIVIAL_BENCHMARK(BM_dladdr_libc_printf, bm_dladdr(printf));
BIONIC_TRIVIAL_BENCHMARK(BM_dladdr_libdl_dladdr, bm_dladdr(dladdr));
BIONIC_TRIVIAL_BENCHMARK(BM_dladdr_local_function, bm_dladdr(local_function));
BIONIC_TRIVIAL_BENCHMARK(BM_dladdr_libbase_split, bm_dladdr(android::base::Split));
