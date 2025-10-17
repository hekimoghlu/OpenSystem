/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
/**
 * @file objc-zalloc.h
 *
 * "zone allocator" for objc.
 *
 * Provides packed allocation for data structures the runtime
 * almost never frees.
 */

#ifndef _OBJC_ZALLOC_H
#define _OBJC_ZALLOC_H

#include <cstdint>
#include <atomic>
#include <cstdlib>

namespace objc {

// Darwin malloc always aligns to 16 bytes
#define MALLOC_ALIGNMENT 16

class AtomicQueue {
#if __LP64__
    using pair_t = __int128_t;
#else
    using pair_t = uint64_t;
#endif
    static constexpr auto relaxed = std::memory_order_relaxed;
    static constexpr auto release = std::memory_order_release;

    struct Entry {
        struct Entry *next;
    };

    union {
        struct {
            Entry        *head;
            unsigned long gen;
        };
        pair_t pair = 0;
    };

    // Can't place the atomic in the union, or we end up needing a constructor
    // which makes _freelist need a static initializer.
    explicit_atomic<pair_t> *atomicPair() {
        return explicit_atomic<pair_t>::from_pointer(&pair);
    }

public:
    void *pop();
    void push_list(void *_head, void *_tail);
    inline void push(void *head)
    {
        push_list(head, head);
    }
};

template<class T, bool useMalloc>
class Zone {
};

template<class T>
class Zone<T, false> {
    struct Element {
        Element *next;
        char buf[sizeof(T) - sizeof(void *)];
    } __attribute__((packed));

    static AtomicQueue _freelist;
    static T *alloc_slow();

public:
    static T *alloc();
    static void free(T *);
};

template<class T>
class Zone<T, true> {
public:
    static inline T *alloc() {
        return reinterpret_cast<T *>(::calloc(sizeof(T), 1));
    }
    static inline void free(T *ptr) {
        ::free(ptr);
    }
};

/*
 * This allocator returns always zeroed memory,
 * and the template needs to be instantiated in objc-zalloc.mm
 */

template<class T>
T *zalloc()
{
    return Zone<T, sizeof(T) % MALLOC_ALIGNMENT == 0>::alloc();
}

template<class T>
void zfree(T *e)
{
    Zone<T, sizeof(T) % MALLOC_ALIGNMENT == 0>::free(e);
}

};

#endif
