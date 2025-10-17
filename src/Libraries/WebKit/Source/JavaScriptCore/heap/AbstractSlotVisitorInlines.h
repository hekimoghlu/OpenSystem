/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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

#include "AbstractSlotVisitor.h"
#include "Heap.h"
#include "WeakInlines.h"
#include "WriteBarrier.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

using ReferrerToken = AbstractSlotVisitor::ReferrerToken;

inline ReferrerToken::ReferrerToken(HeapCell* cell)
    : m_bits(std::bit_cast<uintptr_t>(cell) | HeapCellToken)
{
}

inline ReferrerToken::ReferrerToken(OpaqueRootTag, void* opaqueRoot)
    : m_bits(std::bit_cast<uintptr_t>(opaqueRoot) | OpaqueRootToken)
{
    ASSERT(opaqueRoot);
}

inline ReferrerToken::ReferrerToken(RootMarkReason reason)
    : m_bits((static_cast<uintptr_t>(reason) << tokenTypeShift) | RootMarkReasonToken)
{
}

inline HeapCell* ReferrerToken::asCell() const
{
    return isHeapCell() ? std::bit_cast<HeapCell*>(m_bits & ~tokenTypeMask) : nullptr;
}

inline void* ReferrerToken::asOpaqueRoot() const
{
    return isOpaqueRoot() ? std::bit_cast<HeapCell*>(m_bits & ~tokenTypeMask) : nullptr;
}

inline RootMarkReason ReferrerToken::asRootMarkReason() const
{
    return isRootMarkReason() ? static_cast<RootMarkReason>(m_bits >> tokenTypeShift) : RootMarkReason::None;
}

inline AbstractSlotVisitor::ReferrerContext::ReferrerContext(AbstractSlotVisitor& visitor, ReferrerToken referrer)
    : m_visitor(visitor)
    , m_referrer(referrer)
{
    m_previous = m_visitor.m_context;
    if (m_previous) {
        // An OpaqueRoot contexts can only be on the leaf.
        RELEASE_ASSERT(!m_previous->m_isOpaqueRootContext);
    }
IGNORE_GCC_WARNINGS_BEGIN("dangling-pointer")
    m_visitor.m_context = this;
IGNORE_GCC_WARNINGS_END
}

inline AbstractSlotVisitor::ReferrerContext::ReferrerContext(AbstractSlotVisitor& visitor, AbstractSlotVisitor::OpaqueRootTag)
    : m_visitor(visitor)
    , m_isOpaqueRootContext(true)
{
    m_previous = m_visitor.m_context;
    if (m_previous) {
        // An OpaqueRoot contexts can only be on the leaf.
        RELEASE_ASSERT(!m_previous->m_isOpaqueRootContext);
    }
IGNORE_GCC_WARNINGS_BEGIN("dangling-pointer")
    m_visitor.m_context = this;
IGNORE_GCC_WARNINGS_END
}

inline AbstractSlotVisitor::ReferrerContext::~ReferrerContext()
{
    m_visitor.m_context = m_previous;
}

inline AbstractSlotVisitor::AbstractSlotVisitor(JSC::Heap& heap, CString codeName, ConcurrentPtrHashSet& opaqueRoots)
    : m_heap(heap)
    , m_codeName(codeName)
    , m_opaqueRoots(opaqueRoots)
{
}

inline JSC::Heap* AbstractSlotVisitor::heap() const
{
    return &m_heap;
}

inline VM& AbstractSlotVisitor::vm()
{
    return m_heap.vm();
}

inline const VM& AbstractSlotVisitor::vm() const
{
    return m_heap.vm();
}

inline bool AbstractSlotVisitor::addOpaqueRoot(void* ptr)
{
    if (!ptr)
        return false;
    if (m_ignoreNewOpaqueRoots)
        return false;
    if (!m_opaqueRoots.add(ptr))
        return false;
    if (UNLIKELY(m_needsExtraOpaqueRootHandling))
        didAddOpaqueRoot(ptr);
    m_visitCount++;
    return true;
}

inline bool AbstractSlotVisitor::containsOpaqueRoot(void* ptr) const
{
    bool found = m_opaqueRoots.contains(ptr);
    if (UNLIKELY(found && m_needsExtraOpaqueRootHandling)) {
        auto* nonConstThis = const_cast<AbstractSlotVisitor*>(this);
        nonConstThis->didFindOpaqueRoot(ptr);
    }
    return found;
}

template<typename T>
ALWAYS_INLINE void AbstractSlotVisitor::append(const Weak<T>& weak)
{
    appendUnbarriered(weak.get());
}

template<typename T, typename Traits>
ALWAYS_INLINE void AbstractSlotVisitor::append(const WriteBarrierBase<T, Traits>& slot)
{
    appendUnbarriered(slot.get());
}

template<typename T, typename Traits>
ALWAYS_INLINE void AbstractSlotVisitor::appendHidden(const WriteBarrierBase<T, Traits>& slot)
{
    appendHiddenUnbarriered(slot.get());
}

ALWAYS_INLINE void AbstractSlotVisitor::append(const WriteBarrierStructureID& slot)
{
    appendUnbarriered(reinterpret_cast<JSCell*>(slot.get()));
}

ALWAYS_INLINE void AbstractSlotVisitor::appendHidden(const WriteBarrierStructureID& slot)
{
    appendHiddenUnbarriered(reinterpret_cast<JSCell*>(slot.get()));
}

ALWAYS_INLINE void AbstractSlotVisitor::appendHiddenUnbarriered(JSValue value)
{
    if (value.isCell())
        appendHiddenUnbarriered(value.asCell());
}

template<typename Iterator>
ALWAYS_INLINE void AbstractSlotVisitor::append(Iterator begin, Iterator end)
{
    for (auto it = begin; it != end; ++it)
        append(*it);
}

ALWAYS_INLINE void AbstractSlotVisitor::appendValues(std::span<const WriteBarrier<Unknown, RawValueTraits<Unknown>>> barriers)
{
    for (auto& barrier : barriers)
        append(barrier);
}

ALWAYS_INLINE void AbstractSlotVisitor::appendValues(const WriteBarrierBase<Unknown>* barriers, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        append(barriers[i]);
}

ALWAYS_INLINE void AbstractSlotVisitor::appendValuesHidden(const WriteBarrierBase<Unknown>* barriers, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        appendHidden(barriers[i]);
}

ALWAYS_INLINE void AbstractSlotVisitor::appendUnbarriered(JSValue value)
{
    if (value.isCell())
        appendUnbarriered(value.asCell());
}

ALWAYS_INLINE void AbstractSlotVisitor::appendUnbarriered(JSValue* slot, size_t count)
{
    for (size_t i = count; i--;)
        appendUnbarriered(slot[i]);
}

ALWAYS_INLINE ReferrerToken AbstractSlotVisitor::referrer() const
{
    if (!m_context)
        return nullptr;
    return m_context->referrer();
}

ALWAYS_INLINE void AbstractSlotVisitor::reset()
{
    m_visitCount = 0;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
