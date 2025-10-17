/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#include "WeakBlock.h"

#include "CellContainerInlines.h"
#include "Heap.h"
#include "HeapAnalyzer.h"
#include "JSCInlines.h"
#include "WeakHandleOwner.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(WeakBlock);

WeakBlock* WeakBlock::create(JSC::Heap& heap, CellContainer container)
{
    heap.didAllocateBlock(WeakBlock::blockSize);
    return new (NotNull, WeakBlockMalloc::malloc(blockSize)) WeakBlock(container);

}

void WeakBlock::destroy(JSC::Heap& heap, WeakBlock* block)
{
    block->~WeakBlock();
    WeakBlockMalloc::free(block);
    heap.didFreeBlock(WeakBlock::blockSize);
}

WeakBlock::WeakBlock(CellContainer container)
    : DoublyLinkedListNode<WeakBlock>()
    , m_container(container)
{
    for (size_t i = 0; i < weakImplCount(); ++i) {
        WeakImpl* weakImpl = &weakImpls()[i];
        new (NotNull, weakImpl) WeakImpl;
        addToFreeList(&m_sweepResult.freeList, weakImpl);
    }

    ASSERT(isEmpty());
}

void WeakBlock::lastChanceToFinalize()
{
    for (size_t i = 0; i < weakImplCount(); ++i) {
        WeakImpl* weakImpl = &weakImpls()[i];
        if (weakImpl->state() >= WeakImpl::Finalized)
            continue;
        weakImpl->setState(WeakImpl::Dead);
        finalize(weakImpl);
    }
}

void WeakBlock::sweep()
{
    // If a block is completely empty, a sweep won't have any effect.
    if (isEmpty())
        return;

    SweepResult sweepResult;
    for (size_t i = 0; i < weakImplCount(); ++i) {
        WeakImpl* weakImpl = &weakImpls()[i];
        if (weakImpl->state() == WeakImpl::Dead)
            finalize(weakImpl);
        if (weakImpl->state() == WeakImpl::Deallocated)
            addToFreeList(&sweepResult.freeList, weakImpl);
        else {
            sweepResult.blockIsFree = false;
            if (weakImpl->state() == WeakImpl::Live)
                sweepResult.blockIsLogicallyEmpty = false;
        }
    }

    m_sweepResult = sweepResult;
    ASSERT(!m_sweepResult.isNull());
}

template<typename ContainerType, typename Visitor>
void WeakBlock::specializedVisit(ContainerType& container, Visitor& visitor)
{
    size_t count = weakImplCount();
    HeapAnalyzer* heapAnalyzer = visitor.vm().activeHeapAnalyzer();
    for (size_t i = 0; i < count; ++i) {
        WeakImpl* weakImpl = &weakImpls()[i];
        if (weakImpl->state() != WeakImpl::Live)
            continue;

        WeakHandleOwner* weakHandleOwner = weakImpl->weakHandleOwner();
        if (!weakHandleOwner)
            continue;

        JSValue jsValue = weakImpl->jsValue();
        if (visitor.isMarked(container, jsValue.asCell()))
            continue;
        
        ASCIILiteral reason = ""_s;
        ASCIILiteral* reasonPtr = nullptr;
        if (UNLIKELY(heapAnalyzer))
            reasonPtr = &reason;

        typename Visitor::ReferrerContext context(visitor, Visitor::OpaqueRoot);

        if (!weakHandleOwner->isReachableFromOpaqueRoots(Handle<Unknown>::wrapSlot(&const_cast<JSValue&>(jsValue)), weakImpl->context(), visitor, reasonPtr))
            continue;

        visitor.appendUnbarriered(jsValue);

        if (UNLIKELY(heapAnalyzer)) {
            if (jsValue.isCell())
                heapAnalyzer->setOpaqueRootReachabilityReasonForCell(jsValue.asCell(), *reasonPtr);
        }
    }
}

template<typename Visitor>
ALWAYS_INLINE void WeakBlock::visitImpl(Visitor& visitor)
{
    // If a block is completely empty, a visit won't have any effect.
    if (isEmpty())
        return;

    // If this WeakBlock doesn't belong to a CellContainer, we won't even be here.
    ASSERT(m_container);
    
    if (m_container.isPreciseAllocation())
        specializedVisit(m_container.preciseAllocation(), visitor);
    else
        specializedVisit(m_container.markedBlock(), visitor);
}

void WeakBlock::visit(AbstractSlotVisitor& visitor) { visitImpl(visitor); }
void WeakBlock::visit(SlotVisitor& visitor) { visitImpl(visitor); }

void WeakBlock::reap()
{
    // If a block is completely empty, a reaping won't have any effect.
    if (isEmpty())
        return;

    // If this WeakBlock doesn't belong to a CellContainer, we won't even be here.
    ASSERT(m_container);
    
    HeapVersion markingVersion = m_container.heap()->objectSpace().markingVersion();

    for (size_t i = 0; i < weakImplCount(); ++i) {
        WeakImpl* weakImpl = &weakImpls()[i];
        if (weakImpl->state() > WeakImpl::Dead)
            continue;

        if (m_container.isMarked(markingVersion, weakImpl->jsValue().asCell())) {
            ASSERT(weakImpl->state() == WeakImpl::Live);
            continue;
        }

        weakImpl->setState(WeakImpl::Dead);
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
