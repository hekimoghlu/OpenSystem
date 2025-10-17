/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include "WeakSet.h"

#include "Heap.h"
#include "VM.h"

namespace JSC {

WeakSet::~WeakSet()
{
    if (isOnList())
        remove();
    
    JSC::Heap& heap = *this->heap();
    WeakBlock* next = nullptr;
    for (WeakBlock* block = m_blocks.head(); block; block = next) {
        next = block->next();
        WeakBlock::destroy(heap, block);
    }
    m_blocks.clear();
}

void WeakSet::sweep()
{
    for (WeakBlock* block = m_blocks.head(); block;) {
        heap()->sweepNextLogicallyEmptyWeakBlock();

        WeakBlock* nextBlock = block->next();
        block->sweep();
        if (block->isLogicallyEmptyButNotFree()) {
            // If this WeakBlock is logically empty, but still has Weaks pointing into it,
            // we can't destroy it just yet. Detach it from the WeakSet and hand ownership
            // to the Heap so we don't pin down the entire MarkedBlock or PreciseAllocation.
            m_blocks.remove(block);
            heap()->addLogicallyEmptyWeakBlock(block);
            block->disconnectContainer();
        }
        block = nextBlock;
    }

    resetAllocator();
}

void WeakSet::shrink()
{
    WeakBlock* next;
    for (WeakBlock* block = m_blocks.head(); block; block = next) {
        next = block->next();

        if (block->isEmpty())
            removeAllocator(block);
    }

    resetAllocator();
    
    if (m_blocks.isEmpty() && isOnList())
        remove();
}

WeakBlock::FreeCell* WeakSet::findAllocator(CellContainer container)
{
    if (WeakBlock::FreeCell* allocator = tryFindAllocator())
        return allocator;

    return addAllocator(container);
}

WeakBlock::FreeCell* WeakSet::tryFindAllocator()
{
    while (m_nextAllocator) {
        WeakBlock* block = m_nextAllocator;
        m_nextAllocator = m_nextAllocator->next();

        WeakBlock::SweepResult sweepResult = block->takeSweepResult();
        if (sweepResult.freeList)
            return sweepResult.freeList;
    }

    return nullptr;
}

WeakBlock::FreeCell* WeakSet::addAllocator(CellContainer container)
{
    if (!isOnList())
        heap()->objectSpace().addActiveWeakSet(this);
    
    WeakBlock* block = WeakBlock::create(*heap(), container);
    heap()->didAllocate(WeakBlock::blockSize);
    m_blocks.append(block);
    WeakBlock::SweepResult sweepResult = block->takeSweepResult();
    ASSERT(!sweepResult.isNull() && sweepResult.freeList);
    return sweepResult.freeList;
}

void WeakSet::removeAllocator(WeakBlock* block)
{
    m_blocks.remove(block);
    WeakBlock::destroy(*heap(), block);
}

} // namespace JSC
