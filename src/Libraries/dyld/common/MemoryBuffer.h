/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#ifndef MemoryBuffer_h
#define MemoryBuffer_h

#include <TargetConditionals.h>
#if !TARGET_OS_EXCLAVEKIT

// stl
#include <span>

// Darwin
#include <assert.h>
#include <mach/mach.h>

// common
#include "Defines.h"

// mach-o
#include "Error.h"

class VIS_HIDDEN MemoryBuffer {
protected:
    std::span<uint8_t> buffer;

    MemoryBuffer(std::span<uint8_t> buffer): buffer(buffer) {};
public:

    MemoryBuffer() = default;

    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer(MemoryBuffer&& other) { std::swap(buffer, other.buffer); }

    MemoryBuffer& operator=(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(MemoryBuffer&& other)
    {
        std::swap(buffer, other.buffer);
        return *this;
    }

    std::span<const uint8_t> get() const { return buffer; }
    operator std::span<const uint8_t>() const { return get(); }

    virtual ~MemoryBuffer() = default;
};

class VIS_HIDDEN WritableMemoryBuffer: public MemoryBuffer
{
public:
    WritableMemoryBuffer(std::span<uint8_t> buffer = {}): MemoryBuffer(buffer) {};

    std::span<uint8_t> get() const { return buffer; }
    operator std::span<uint8_t>() const { return get(); }

    static WritableMemoryBuffer allocate(size_t size);
};

class VIS_HIDDEN VMMemoryBuffer: public WritableMemoryBuffer
{
    VMMemoryBuffer(std::span<uint8_t> buffer): WritableMemoryBuffer(buffer) {};

public:

    static WritableMemoryBuffer allocate(size_t size)
    {
        vm_address_t addr;
        kern_return_t ret = ::vm_allocate(mach_task_self(), &addr, size, VM_FLAGS_ANYWHERE);
        if ( ret != KERN_SUCCESS ) {
            assert(false && "couldn't allocate memory");
            return WritableMemoryBuffer();
        }

        return VMMemoryBuffer({(uint8_t*)addr, size});
    }

    ~VMMemoryBuffer()
    {
        if ( buffer.data() != nullptr ) {
            vm_deallocate(mach_task_self(), (vm_address_t)buffer.data(), buffer.size());
            buffer = {};
        }
    }
};

inline WritableMemoryBuffer WritableMemoryBuffer::allocate(size_t size)
{
    return VMMemoryBuffer::allocate(size);
}

#endif // !TARGET_OS_EXCLAVEKIT

#endif /* MemoryBuffer_h */
