/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#include "CPUCount.h"
#include "Memory.h"
#include "balloon.h"
#include <array>
#include <chrono>
#include <memory>
#include <stddef.h>
#include <strings.h>

#include "mbmalloc.h"

void benchmark_balloon(CommandLine&)
{
    const size_t chunkSize = 1 * 1024;
    const size_t balloonSize = 100 * 1024 * 1024;
    const size_t steadySize = 10 * 1024 * 1024;
    
    std::array<void*, balloonSize / chunkSize> balloon;
    std::array<void*, balloonSize / chunkSize> steady;

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < balloon.size(); ++i) {
        balloon[i] = mbmalloc(chunkSize);
        bzero(balloon[i], chunkSize);
    }

    for (size_t i = 0; i < balloon.size(); ++i)
        mbfree(balloon[i], chunkSize);

    auto stop = std::chrono::steady_clock::now();
    
    auto benchmarkTime = stop - start;

    start = std::chrono::steady_clock::now();

    // Converts bytes to time -- for reporting's sake -- by waiting a while until
    // the heap shrinks back down. This isn't great for pooling with other
    // benchmarks in a geometric mean of throughput, but it's OK for basic testing.
    while (currentMemoryBytes().resident > 2 * steadySize
        && std::chrono::steady_clock::now() - start < 8 * benchmarkTime) {
        for (size_t i = 0; i < steady.size(); ++i) {
            steady[i] = mbmalloc(chunkSize);
            bzero(steady[i], chunkSize);
        }

        for (size_t i = 0; i < steady.size(); ++i)
            mbfree(steady[i], chunkSize);
    }
}
