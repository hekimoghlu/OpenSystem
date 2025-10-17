/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

#if ENABLE(C_LOOP)

#include "Register.h"
#include <wtf/Noncopyable.h>
#include <wtf/PageReservation.h>

namespace JSC {

    class CodeBlockSet;
    class ConservativeRoots;
    class JITStubRoutineSet;
    class VM;
    class LLIntOffsetsExtractor;

    class CLoopStack {
        WTF_MAKE_NONCOPYABLE(CLoopStack);
    public:
        // Allow 8k of excess registers before we start trying to reap the stack
        static constexpr ptrdiff_t maxExcessCapacity = 8 * 1024;

        CLoopStack(VM&);
        ~CLoopStack();
        
        bool ensureCapacityFor(Register* newTopOfStack);

        bool containsAddress(Register* address) { return (lowAddress() <= address && address < highAddress()); }
        static size_t committedByteCount();

        void gatherConservativeRoots(ConservativeRoots&, JITStubRoutineSet&, CodeBlockSet&);
        void sanitizeStack();

        inline void* currentStackPointer() const;
        void setCurrentStackPointer(void* sp) { m_currentStackPointer = sp; }

        size_t size() const { return highAddress() - lowAddress(); }

        void setSoftReservedZoneSize(size_t);
        bool isSafeToRecurse() const;

    private:
        Register* lowAddress() const
        {
            return m_end;
        }

        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

        Register* highAddress() const
        {
            return reinterpret_cast_ptr<Register*>(static_cast<char*>(m_reservation.base()) + m_reservation.size());
        }

        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

        Register* reservationTop() const
        {
            char* reservationTop = static_cast<char*>(m_reservation.base());
            return reinterpret_cast_ptr<Register*>(reservationTop);
        }

        bool grow(Register* newTopOfStack);
        void releaseExcessCapacity();
        void addToCommittedByteCount(long);

        void setCLoopStackLimit(Register* newTopOfStack);

        VM& m_vm;
        CallFrame*& m_topCallFrame;

        // The following is always true:
        //    reservationTop() <= m_commitTop <= m_end <= m_currentStackPointer <= highAddress()
        Register* m_end; // lowest address of JS allocatable stack memory.
        Register* m_commitTop; // lowest address of committed memory.
        PageReservation m_reservation;
        void* m_lastStackPointer;
        void* m_currentStackPointer;
        ptrdiff_t m_softReservedZoneSizeInRegisters;

        friend class LLIntOffsetsExtractor;
    };

} // namespace JSC

#endif // ENABLE(C_LOOP)
