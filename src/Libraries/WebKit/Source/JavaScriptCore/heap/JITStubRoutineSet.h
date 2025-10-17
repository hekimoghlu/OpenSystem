/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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

#include "JITStubRoutine.h"
#include <wtf/HashMap.h>
#include <wtf/Range.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

using WTF::Range;

namespace JSC {

class GCAwareJITStubRoutine;
class VM;

#if ENABLE(JIT)

class JITStubRoutineSet {
    WTF_MAKE_NONCOPYABLE(JITStubRoutineSet);
    WTF_MAKE_TZONE_ALLOCATED(JITStubRoutineSet);
public:
    JITStubRoutineSet();
    ~JITStubRoutineSet();
    
    void add(GCAwareJITStubRoutine*);

    void clearMarks();
    
    void mark(void* candidateAddress)
    {
        uintptr_t address = removeCodePtrTag<uintptr_t>(candidateAddress);
        if (!m_range.contains(address))
            return;
        markSlow(address);
    }

    void prepareForConservativeScan();
    
    void deleteUnmarkedJettisonedStubRoutines(VM&);

    template<typename Visitor> void traceMarkedStubRoutines(Visitor&);
    
private:
    void markSlow(uintptr_t address);
    
    struct Routine {
        uintptr_t startAddress;
        GCAwareJITStubRoutine* routine;
    };
    Vector<Routine> m_routines;
    Vector<GCAwareJITStubRoutine*> m_immutableCodeRoutines;
    Range<uintptr_t> m_range { 0, 0 };
};

#else // !ENABLE(JIT)

class JITStubRoutineSet {
    WTF_MAKE_NONCOPYABLE(JITStubRoutineSet);
    WTF_MAKE_TZONE_ALLOCATED(JITStubRoutineSet);
public:
    JITStubRoutineSet() { }
    ~JITStubRoutineSet() { }

    void add(GCAwareJITStubRoutine*) { }
    void clearMarks() { }
    void mark(void*) { }
    void prepareForConservativeScan() { }
    void deleteUnmarkedJettisonedStubRoutines(VM&) { }
    template<typename Visitor> void traceMarkedStubRoutines(Visitor&) { }
};

#endif // !ENABLE(JIT)

} // namespace JSC
