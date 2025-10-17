/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

#if ENABLE(B3_JIT)

#include "AirLiveness.h"
#include "AirTmpMap.h"
#include <wtf/Nonmovable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { 

class CCallHelpers;

namespace B3 { namespace Air {

class Code;

class GenerateAndAllocateRegisters {
    WTF_MAKE_TZONE_ALLOCATED(GenerateAndAllocateRegisters);
    WTF_MAKE_NONMOVABLE(GenerateAndAllocateRegisters);

    struct TmpData {
        StackSlot* spillSlot { nullptr };
        Reg reg;
    };

public:
    GenerateAndAllocateRegisters(Code&);

    void prepareForGeneration();
    void generate(CCallHelpers&);

private:
    void insertBlocksForFlushAfterTerminalPatchpoints();
    void release(Tmp, Reg);
    void flush(Tmp, Reg);
    void spill(Tmp, Reg);
    void alloc(Tmp, Reg, Arg::Role);
    void freeDeadTmpsAtCurrentInst();
    void freeDeadTmpsAtCurrentBlock();
    bool assignTmp(Tmp&, Bank, Arg::Role);
    void buildLiveRanges(UnifiedTmpLiveness&);
    bool isDisallowedRegister(Reg);

    void checkConsistency();

    Code& m_code;
    CCallHelpers* m_jit { nullptr };

    TmpMap<TmpData> m_map;

#if ASSERT_ENABLED
    Vector<Tmp> m_allTmps[numBanks];
#endif

    Vector<Reg> m_registers[numBanks];
    ScalarRegisterSet m_availableRegs[numBanks];
    size_t m_globalInstIndex;
    IndexMap<Reg, Tmp>* m_currentAllocation { nullptr };
    TmpMap<size_t> m_liveRangeEnd;
    UncheckedKeyHashMap<size_t, Vector<Tmp, 2>> m_tmpsToRelease;
    RegisterSet m_namedUsedRegs;
    RegisterSet m_namedDefdRegs;
    RegisterSetBuilder m_earlyClobber;
    RegisterSetBuilder m_lateClobber;
    RegisterSetBuilder m_clobberedToClear;
    RegisterSet m_allowedRegisters;
    std::unique_ptr<UnifiedTmpLiveness> m_liveness;

    struct PatchSpillData {
        MacroAssembler::Jump jump;
        MacroAssembler::Label continueLabel;
        UncheckedKeyHashMap<Tmp, Arg*> defdTmps;
    };

    UncheckedKeyHashMap<BasicBlock*, PatchSpillData> m_blocksAfterTerminalPatchForSpilling;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
