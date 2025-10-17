/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#include <wtf/PrintStream.h>
#include <wtf/SinglyLinkedListWithTail.h>

namespace JSC {

class BlockDirectory;
class Heap;
class Subspace;

class AlignedMemoryAllocator {
    WTF_MAKE_NONCOPYABLE(AlignedMemoryAllocator);
    WTF_MAKE_FAST_ALLOCATED;
public:
    AlignedMemoryAllocator();
    virtual ~AlignedMemoryAllocator();
    
    virtual void* tryAllocateAlignedMemory(size_t alignment, size_t size) = 0;
    virtual void freeAlignedMemory(void*) = 0;
    
    // This can't be pure virtual as it breaks our Dumpable concept.
    // FIXME: Make this virtual after we stop suppporting the Montery Clang.
    virtual void dump(PrintStream&) const { }

    void registerDirectory(Heap&, BlockDirectory*);
    BlockDirectory* firstDirectory() const { return m_directories.first(); }

    void registerSubspace(Subspace*);

    // Some of derived memory allocators do not have these features because they do not use them.
    // For example, IsoAlignedMemoryAllocator does not have "realloc" feature since it never extends / shrinks the allocated memory region.
    virtual void* tryAllocateMemory(size_t) = 0;
    virtual void freeMemory(void*) = 0;
    virtual void* tryReallocateMemory(void*, size_t) = 0;

private:
    SinglyLinkedListWithTail<BlockDirectory> m_directories;
    SinglyLinkedListWithTail<Subspace> m_subspaces;
};

} // namespace WTF

