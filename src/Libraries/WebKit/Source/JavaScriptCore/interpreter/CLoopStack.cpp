/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
#include "CLoopStack.h"

#if ENABLE(C_LOOP)

#include "CLoopStackInlines.h"
#include "ConservativeRoots.h"
#include "Interpreter.h"
#include "JSCInlines.h"
#include "Options.h"
#include <wtf/Lock.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

static size_t committedBytesCount = 0;

static size_t commitSize()
{
    static size_t size = std::max<size_t>(16 * 1024, pageSize());
    return size;
}

static Lock stackStatisticsMutex;

CLoopStack::CLoopStack(VM& vm)
    : m_vm(vm)
    , m_topCallFrame(vm.topCallFrame)
    , m_softReservedZoneSizeInRegisters(0)
{
    size_t capacity = Options::maxPerThreadStackUsage();
    capacity = WTF::roundUpToMultipleOf(pageSize(), capacity);
    ASSERT(capacity && isPageAligned(capacity));

    m_reservation = PageReservation::reserve(WTF::roundUpToMultipleOf(commitSize(), capacity), OSAllocator::UnknownUsage);

    auto* bottomOfStack = highAddress();
    setCLoopStackLimit(bottomOfStack);
    ASSERT(m_end == bottomOfStack);
    m_commitTop = bottomOfStack;
    m_lastStackPointer = bottomOfStack;
    m_currentStackPointer = bottomOfStack;

    m_topCallFrame = 0;
}

CLoopStack::~CLoopStack()
{
    ptrdiff_t sizeToDecommit = reinterpret_cast<char*>(highAddress()) - reinterpret_cast<char*>(m_commitTop);
    m_reservation.decommit(reinterpret_cast<void*>(m_commitTop), sizeToDecommit);
    addToCommittedByteCount(-sizeToDecommit);
    m_reservation.deallocate();
}

bool CLoopStack::grow(Register* newTopOfStack)
{
    Register* newTopOfStackWithReservedZone = newTopOfStack - m_softReservedZoneSizeInRegisters;

    // If we have already committed enough memory to satisfy this request,
    // just update the end pointer and return.
    if (newTopOfStackWithReservedZone >= m_commitTop) {
        setCLoopStackLimit(newTopOfStack);
        return true;
    }

    // Compute the chunk size of additional memory to commit, and see if we
    // have it still within our budget. If not, we'll fail to grow and
    // return false.
    ptrdiff_t delta = reinterpret_cast<char*>(m_commitTop) - reinterpret_cast<char*>(newTopOfStackWithReservedZone);
    delta = WTF::roundUpToMultipleOf(commitSize(), delta);
    Register* newCommitTop = m_commitTop - (delta / sizeof(Register));
    if (newCommitTop < reservationTop())
        return false;

    // Otherwise, the growth is still within our budget. Commit it and return true.
    m_reservation.commit(newCommitTop, delta);
    addToCommittedByteCount(delta);
    m_commitTop = newCommitTop;
    newTopOfStack = m_commitTop + m_softReservedZoneSizeInRegisters;
    setCLoopStackLimit(newTopOfStack);
    return true;
}

void CLoopStack::gatherConservativeRoots(ConservativeRoots& conservativeRoots, JITStubRoutineSet& jitStubRoutines, CodeBlockSet& codeBlocks)
{
    conservativeRoots.add(currentStackPointer(), highAddress(), jitStubRoutines, codeBlocks);
}

void CLoopStack::sanitizeStack()
{
#if !ASAN_ENABLED
    void* stackTop = currentStackPointer();
    ASSERT(stackTop <= highAddress());
    if (m_lastStackPointer < stackTop) {
        char* begin = reinterpret_cast<char*>(m_lastStackPointer);
        char* end = reinterpret_cast<char*>(stackTop);
        memset(begin, 0, end - begin);
    }
    
    m_lastStackPointer = stackTop;
#endif
}

void CLoopStack::releaseExcessCapacity()
{
    Register* highAddressWithReservedZone = highAddress() - m_softReservedZoneSizeInRegisters;
    ptrdiff_t delta = reinterpret_cast<char*>(highAddressWithReservedZone) - reinterpret_cast<char*>(m_commitTop);
    m_reservation.decommit(m_commitTop, delta);
    addToCommittedByteCount(-delta);
    m_commitTop = highAddressWithReservedZone;
}

void CLoopStack::addToCommittedByteCount(long byteCount)
{
    Locker locker { stackStatisticsMutex };
    ASSERT(static_cast<long>(committedBytesCount) + byteCount > -1);
    committedBytesCount += byteCount;
}

void CLoopStack::setSoftReservedZoneSize(size_t reservedZoneSize)
{
    m_softReservedZoneSizeInRegisters = reservedZoneSize / sizeof(Register);
    if (m_commitTop > m_end - m_softReservedZoneSizeInRegisters)
        grow(m_end);
}

bool CLoopStack::isSafeToRecurse() const
{
    void* reservationLimit = reinterpret_cast<int8_t*>(reservationTop() + m_softReservedZoneSizeInRegisters);
    return !m_topCallFrame || (m_topCallFrame->topOfFrame() > reservationLimit);
}

size_t CLoopStack::committedByteCount()
{
    Locker locker { stackStatisticsMutex };
    return committedBytesCount;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(C_LOOP)
