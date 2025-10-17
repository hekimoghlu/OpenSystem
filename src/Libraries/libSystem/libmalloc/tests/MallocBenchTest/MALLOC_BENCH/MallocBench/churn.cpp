/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#include "churn.h"
#include <memory>
#include <stddef.h>

#include "mbmalloc.h"

struct HeapDouble {
    void* operator new(size_t size) { return mbmalloc(size); }
    void operator delete(void* p, size_t size) { mbfree(p, size); }

    HeapDouble(double d) : value(d) { }
    const HeapDouble& operator+=(const HeapDouble& other) { value += other.value; return *this; }
    double value;
};

void benchmark_churn(CommandLine& commandLine)
{
    size_t times = 7000000;
    if (commandLine.isParallel())
        times /= cpuCount();

    auto total = std::unique_ptr<HeapDouble>(new HeapDouble(0.0));
    for (size_t i = 0; i < times; ++i) {
        auto heapDouble = std::unique_ptr<HeapDouble>(new HeapDouble(i));
        *total += *heapDouble;
    }
}
