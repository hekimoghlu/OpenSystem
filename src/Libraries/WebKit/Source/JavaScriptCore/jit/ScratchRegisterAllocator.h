/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#if ENABLE(JIT)

#include "FPRInfo.h"
#include "RegisterSet.h"

namespace JSC {

class AssemblyHelpers;
struct ScratchBuffer;

// This class provides a low-level register allocator for use in stubs.

class ScratchRegisterAllocator {
public:
    ScratchRegisterAllocator() = default;
    ScratchRegisterAllocator(const RegisterSet& usedRegisters);
    ~ScratchRegisterAllocator();

    void lock(GPRReg);
    void lock(FPRReg);
    void lock(JSValueRegs);
    
    template<typename BankInfo>
    typename BankInfo::RegisterType allocateScratch();
    
    GPRReg allocateScratchGPR();
    FPRReg allocateScratchFPR();
    
    bool didReuseRegisters() const
    {
        return !!m_numberOfReusedRegisters;
    }
    
    unsigned numberOfReusedRegisters() const
    {
        return m_numberOfReusedRegisters;
    }

    RegisterSet usedRegisters() const { return m_usedRegisters; }
    
    enum class ExtraStackSpace { SpaceForCCall, NoExtraSpace };

    struct PreservedState {
        PreservedState()
            : numberOfBytesPreserved(std::numeric_limits<unsigned>::max())
            , extraStackSpaceRequirement(ExtraStackSpace::SpaceForCCall)
        { }

        PreservedState(unsigned numberOfBytes, ExtraStackSpace extraStackSpace)
            : numberOfBytesPreserved(numberOfBytes)
            , extraStackSpaceRequirement(extraStackSpace)
        { }

        explicit operator bool() const { return numberOfBytesPreserved != std::numeric_limits<unsigned>::max(); }

        unsigned numberOfBytesPreserved;
        ExtraStackSpace extraStackSpaceRequirement;
    };

    PreservedState preserveReusedRegistersByPushing(AssemblyHelpers& jit, ExtraStackSpace);
    void restoreReusedRegistersByPopping(AssemblyHelpers& jit, const PreservedState&);

    static unsigned preserveRegistersToStackForCall(AssemblyHelpers& jit, const RegisterSet& usedRegisters, unsigned extraPaddingInBytes);
    static void restoreRegistersFromStackForCall(AssemblyHelpers& jit, const RegisterSet& usedRegisters, const RegisterSet& ignore, unsigned numberOfStackBytesUsedForRegisterPreservation, unsigned extraPaddingInBytes);

private:
    RegisterSet m_usedRegisters { };
    RegisterSet m_scratchRegisters { };
    ScalarRegisterSet m_lockedRegisters { };
    unsigned m_numberOfReusedRegisters { 0 };
};

} // namespace JSC

#endif // ENABLE(JIT)
