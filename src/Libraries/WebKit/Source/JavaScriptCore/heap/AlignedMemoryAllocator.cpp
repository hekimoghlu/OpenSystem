/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#include "config.h"
#include "AlignedMemoryAllocator.h"

#include "BlockDirectory.h"
#include "HeapInlines.h"
#include "Subspace.h"

namespace JSC { 

AlignedMemoryAllocator::AlignedMemoryAllocator() = default;

AlignedMemoryAllocator::~AlignedMemoryAllocator() = default;

void AlignedMemoryAllocator::registerDirectory(JSC::Heap& heap, BlockDirectory* directory)
{
    RELEASE_ASSERT(!directory->nextDirectoryInAlignedMemoryAllocator());
    
    if (m_directories.isEmpty()) {
        ASSERT_UNUSED(heap, !Thread::mayBeGCThread() || heap.worldIsStopped());
        for (Subspace* subspace = m_subspaces.first(); subspace; subspace = subspace->nextSubspaceInAlignedMemoryAllocator())
            subspace->didCreateFirstDirectory(directory);
    }
    
    m_directories.append(std::mem_fn(&BlockDirectory::setNextDirectoryInAlignedMemoryAllocator), directory);
}

void AlignedMemoryAllocator::registerSubspace(Subspace* subspace)
{
    RELEASE_ASSERT(!subspace->nextSubspaceInAlignedMemoryAllocator());
    m_subspaces.append(std::mem_fn(&Subspace::setNextSubspaceInAlignedMemoryAllocator), subspace);
}

} // namespace JSC


