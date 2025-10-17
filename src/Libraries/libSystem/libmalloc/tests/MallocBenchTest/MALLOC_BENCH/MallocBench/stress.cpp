/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "Benchmark.h"
#include "CPUCount.h"
#include "stress.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stddef.h>
#include <vector>

#include "mbmalloc.h"

static const size_t kB = 1024;
static const size_t MB = kB * kB;

struct Object {
    Object(void* pointer, size_t size, long uuid)
        : pointer(pointer)
        , size(size)
        , uuid(uuid)
    {
    }

    void* pointer;
    size_t size;
    long uuid;
};

class SizeStream {
public:
    SizeStream()
        : m_state(Small)
        , m_count(0)
    {
    }

    size_t next()
    {
        switch (m_state) {
        case Small: {
            if (++m_count == smallCount) {
                m_state = Medium;
                m_count = 0;
            }
            return random() % smallMax;
        }
                
        case Medium: {
            if (++m_count == mediumCount) {
                m_state = Large;
                m_count = 0;
            }
            return random() % mediumMax;
        }
                
        case Large: {
            if (++m_count == largeCount) {
                m_state = Small;
                m_count = 0;
            }
            return random() % largeMax;
        }
        }
        assert(0);
        return 0;
    }

private:
    static const size_t smallCount = 1000;
    static const size_t smallMax = 16 * kB;

    static const size_t mediumCount = 100;
    static const size_t mediumMax = 512 * kB;
    
    static const size_t largeCount = 10;
    static const size_t largeMax = 4 * MB;

    enum { Small, Medium, Large } m_state;
    size_t m_count;
};

Object allocate(size_t size)
{
    Object object(mbmalloc(size), size, random());
    for (size_t i = 0; i < size / sizeof(long); ++i)
        (static_cast<long*>(object.pointer))[i] = object.uuid;
    return object;
}

void deallocate(const Object& object)
{
    for (size_t i = 0; i < object.size / sizeof(long); ++i) {
        if ((static_cast<long*>(object.pointer))[i] != object.uuid)
            abort();
    }

    mbfree(object.pointer, object.size);
}

void benchmark_stress(CommandLine&)
{
    const size_t heapSize = 100 * MB;
    const size_t churnSize = .05 * heapSize;
    const size_t churnCount = 100;
    
    srandom(1); // For consistency between runs.

    std::vector<Object> objects;
    
    SizeStream sizeStream;
    
    size_t size = 0;
    for (size_t remaining = heapSize; remaining; remaining -= std::min(remaining, size)) {
        size = sizeStream.next();
        objects.push_back(allocate(size));
    }
    
    for (size_t i = 0; i < churnCount; ++i) {
        std::vector<Object> objectsToFree;
        for (size_t remaining = churnSize; remaining; remaining -= std::min(remaining, size)) {
            size = sizeStream.next();
            Object object = allocate(size);

            size_t index = random() % objects.size();
            objectsToFree.push_back(objects[index]);
            objects[index] = object;
        }

        for (auto& object : objectsToFree)
            deallocate(object);
        
        mbscavenge();
    }
    
    for (auto& object : objects)
        mbfree(object.pointer, object.size);
}
