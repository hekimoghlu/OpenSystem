/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#pragma once

#include <wtf/Assertions.h>
#include <wtf/CodePtr.h>
#include <wtf/RedBlackTree.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {

class MetaAllocator;
class PrintStream;

DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(MetaAllocatorHandle);
class MetaAllocatorHandle : public ThreadSafeRefCounted<MetaAllocatorHandle>, public RedBlackTree<MetaAllocatorHandle, void*>::Node {
    WTF_MAKE_FAST_COMPACT_ALLOCATED_WITH_HEAP_IDENTIFIER(MetaAllocatorHandle);

public:
    using MemoryPtr = CodePtr<HandleMemoryPtrTag>;

    WTF_EXPORT_PRIVATE ~MetaAllocatorHandle();

    MemoryPtr start() const
    {
        return m_start;
    }

    MemoryPtr end() const
    {
        return m_end;
    }

    uintptr_t startAsInteger() const
    {
        return m_start.untaggedPtr<uintptr_t>();
    }

    uintptr_t endAsInteger() const
    {
        return m_end.untaggedPtr<uintptr_t>();
    }

    size_t sizeInBytes() const
    {
        return m_end.untaggedPtr<size_t>() - m_start.untaggedPtr<size_t>();
    }
    
    bool containsIntegerAddress(uintptr_t address) const
    {
        return address >= startAsInteger() && address < endAsInteger();
    }
    
    bool contains(void* address) const
    {
        return containsIntegerAddress(reinterpret_cast<uintptr_t>(address));
    }
        
    WTF_EXPORT_PRIVATE void shrink(size_t newSizeInBytes);
    
    MetaAllocator& allocator()
    {
        return m_allocator;
    }

    void* key()
    {
        return m_start.untaggedPtr();
    }

    WTF_EXPORT_PRIVATE void dump(PrintStream& out) const;
    
private:
    MetaAllocatorHandle(MetaAllocator&, MemoryPtr start, size_t sizeInBytes);

    MetaAllocator& m_allocator;
    MemoryPtr m_start;
    MemoryPtr m_end;

    friend class MetaAllocator;
};

}
