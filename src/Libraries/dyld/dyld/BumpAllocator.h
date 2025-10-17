/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#ifndef BumpAllocator_h
#define BumpAllocator_h

#include <stdint.h>
#include <stddef.h>

namespace dyld4 {

class BumpAllocator
{
public:
                BumpAllocator() { }
                ~BumpAllocator();

    void        append(const void* payload, uint64_t payloadSize);
    void        zeroFill(uint64_t payloadSize);
    void        align(unsigned multipleOf);
    uint64_t    size() const { return _usageEnd - _vmAllocationStart; }
    const void* finalize();

private:
    template <typename T>
    friend class BumpAllocatorPtr;
    uint8_t*    start() { return _vmAllocationStart; }

protected:
    uint8_t* _vmAllocationStart = nullptr;
    uint64_t _vmAllocationSize  = 0;
    uint8_t* _usageEnd          = nullptr;
};

// Gives a safe pointer in to a BumpAllocator.  This pointer is safe to use across
// appends to the allocator which might change the address of the allocated memory.
template <typename T>
class BumpAllocatorPtr
{
public:
    BumpAllocatorPtr(BumpAllocator& allocator, uint64_t offset)
        : _allocator(allocator)
        , _offset(offset)
    {
    }

    T* get() const
    {
        return (T*)(_allocator.start() + _offset);
    }

    T* operator->() const
    {
        return get();
    }

private:
    BumpAllocator& _allocator;
    uint64_t      _offset;
};

} // namespace dyld4

#endif // BumpAllocator_h


