/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
#include "medium.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <strings.h>

#include "mbmalloc.h"

using namespace std;

struct Object {
    double* p;
    size_t size;
};

void benchmark_medium(CommandLine& commandLine)
{
    size_t times = 1;

    size_t vmSize = 1ul * 1024 * 1024 * 1024;
    size_t objectSizeMin = 1 * 1024;
    size_t objectSizeMax = 8 * 1024;
    if (commandLine.isParallel())
        vmSize /= cpuCount();

    size_t objectCount = vmSize / objectSizeMin;

    srandom(0); // For consistency between runs.

    for (size_t i = 0; i < times; ++i) {
        Object* objects = (Object*)mbmalloc(objectCount * sizeof(Object));
        bzero(objects, objectCount * sizeof(Object));

        for (size_t i = 0, remaining = vmSize; remaining > objectSizeMin; ++i) {
            size_t size = min<size_t>(remaining, max<size_t>(objectSizeMin, random() % objectSizeMax));
            objects[i] = { (double*)mbmalloc(size), size };
            bzero(objects[i].p, size);
            remaining -= size;
        }

        for (size_t i = 0; i < objectCount && objects[i].p; ++i)
            mbfree(objects[i].p, objects[i].size);

        mbfree(objects, objectCount * sizeof(Object));
    }
}
