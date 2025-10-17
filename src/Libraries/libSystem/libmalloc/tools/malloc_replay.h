/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#ifndef __MALLOC_REPLAY_H
#define __MALLOC_REPLAY_H

#include "trace.h"
#include <vector>

//
//Our file format
//

// Definitions for the event chunk.
#define MALLOC_EVENTS_TAG	(uint32_t)0xe001e001
#define MALLOC_EVENTS_V_MAJOR	1
#define MALLOC_EVENTS_V_MINOR	1

enum operation {
    op_malloc = 0x01,
    op_free = 0x02,
    op_realloc = 0x03,
    op_memalign = 0x04,
    op_calloc = 0x05,
    op_valloc = 0x06,
};

static const int operation_count = op_valloc;
static const char *mcall_names[] = {"malloc", "free", "realloc", "memalign", "calloc", "valloc"};

static inline const char *
mcall_to_name(int call_num) {
	if (call_num > 0 && call_num <= operation_count) {
		return mcall_names[call_num - 1];
	}
	return NULL;
}

enum flags {
    flag_stacks     = 0x00000001,
    flag_timestamps = 0x00000002
};

struct compressed_header {
    uint16_t version;
    uint64_t flags;
} __attribute__((packed));

struct compressed_operation {
	uint8_t opcode;
	uint8_t core;
	uint32_t body[];
}__attribute__((packed));

struct compressed_alloc {
    uint64_t address;
    uint32_t size;
} __attribute__((packed));

struct compressed_calloc {
    uint64_t address;
    uint32_t count;
    uint32_t size;
} __attribute__((packed));

struct compressed_memalign {
    uint64_t address;
    uint32_t alignment;
    uint32_t size;
} __attribute__((packed));

struct compressed_free {
    uint64_t address;
} __attribute__((packed));

struct compressed_realloc {
    uint64_t oldAddress;
    uint64_t newAddress;
    uint32_t size;
} __attribute__((packed));

struct compressed_stack_key {
    uint64_t stackKey;
} __attribute__((packed));

struct compressed_time {
    uint64_t timestamp;
} __attribute__((packed));


//
//Our allocator to allocate from a specific zone.
//
extern malloc_zone_t* s_zone;

template <class T>
class ReplayAllocator {
public:
    // type definitions
    typedef T        value_type;
    typedef T*       pointer;
    typedef const T* const_pointer;
    typedef T&       reference;
    typedef const T& const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // rebind allocator to type U
    template <class U>
    struct rebind {
        typedef ReplayAllocator<U> other;
    };

    // return address of values
    pointer address (reference value) const
	{
        return &value;
    }
    const_pointer address (const_reference value) const
	{
        return &value;
    }

    /* constructors and destructor
     * - nothing to do because the allocator has no state
     */
    ReplayAllocator() throw() { }
    ReplayAllocator(const ReplayAllocator&) throw() { }
    template <class U>
    ReplayAllocator (const ReplayAllocator<U>&) throw() { }
    ~ReplayAllocator() throw() { }

    // return maximum number of elements that can be allocated
    size_type max_size () const throw()
	{
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    // allocate but don't initialize num elements of type T
    pointer allocate (size_type num, const void* = 0)
	{
        return (pointer)malloc_zone_malloc(s_zone, num * sizeof(T));
    }

    // initialize elements of allocated storage p with value value
    void construct (pointer p, const T& value)
	{
        // initialize memory with placement new
        new((void*)p)T(value);
    }

    // destroy elements of initialized storage p
    void destroy (pointer p)
	{
        // destroy objects by calling their destructor
        p->~T();
    }

    // deallocate storage p of deleted elements
    void deallocate (pointer p, size_type num)
	{
        malloc_zone_free(s_zone, p);
    }
};

template <class T1, class T2>
bool operator== (const ReplayAllocator<T1>&,
                 const ReplayAllocator<T2>&) throw()
{
    return true;
}
template <class T1, class T2>
bool operator!= (const ReplayAllocator<T1>&,
                 const ReplayAllocator<T2>&) throw()
{
    return false;
}

typedef struct replay_malloc_magazine {
    uint64_t baseAddress;
    uint64_t extent;
    uint32_t pages_dirty;
} *replay_malloc_magazine_t;

typedef struct replay_malloc_zone {
    const char* name;
    std::vector<replay_malloc_magazine, ReplayAllocator<replay_malloc_magazine> > magazines;
} *replay_malloc_zone_t;


#endif // __MALLOC_REPLAY_H
