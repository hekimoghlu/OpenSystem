/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#if ENABLE(FTL_JIT)

#include "B3ValueRep.h"
#include "CodeOrigin.h"
#include "DFGExitProfile.h"
#include "DFGNodeOrigin.h"
#include "DFGOSRExitBase.h"
#include "FTLAbbreviatedTypes.h"
#include "FTLExitTimeObjectMaterialization.h"
#include "FTLExitValue.h"
#include "FTLFormattedValue.h"
#include "FTLOSRExitHandle.h"
#include "FTLStackmapArgumentList.h"
#include "HandlerInfo.h"
#include "MethodOfGettingAValueProfile.h"
#include "Operands.h"
#include "Reg.h"
#include "ValueProfile.h"
#include "VirtualRegister.h"
#include <wtf/FixedVector.h>

namespace JSC {

class TrackedReferences;

namespace B3 {
class StackmapGenerationParams;
namespace Air {
struct GenerationContext;
} // namespace Air
} // namespace B3

namespace DFG {
struct NodeOrigin;
} // namespace DFG;

namespace FTL {

class State;
struct OSRExitDescriptorImpl;
struct OSRExitHandle;

struct OSRExitDescriptor {
private:
    WTF_MAKE_NONCOPYABLE(OSRExitDescriptor);
public:
    OSRExitDescriptor(
        DataFormat profileDataFormat, MethodOfGettingAValueProfile,
        unsigned numberOfArguments, unsigned numberOfLocals, unsigned numberOfTmps);

    // The first argument to the exit call may be a value we wish to profile.
    // If that's the case, the format will be not Invalid and we'll have a
    // method of getting a value profile. Note that all of the ExitArgument's
    // are already aware of this possible off-by-one, so there is no need to
    // correct them.
    DataFormat m_profileDataFormat;
    MethodOfGettingAValueProfile m_valueProfile;
    
    FixedOperands<ExitValue> m_values;
    Bag<ExitTimeObjectMaterialization> m_materializations;

    void validateReferences(const TrackedReferences&);

    // Call this once we have a place to emit the OSR exit jump and we have data about how the state
    // should be recovered. This effectively emits code that does the exit, though the code is really a
    // patchable jump and we emit the real code lazily. The description of how to emit the real code is
    // up to the OSRExit object, which this creates. Note that it's OK to drop the OSRExitHandle object
    // on the ground. It contains information that is mostly not useful if you use this API, since after
    // this call, the OSRExit is simply ready to go.
    Ref<OSRExitHandle> emitOSRExit(
        State&, ExitKind, const DFG::NodeOrigin&, CCallHelpers&, const B3::StackmapGenerationParams&,
        uint32_t dfgNodeIndex, unsigned offset);

    // In some cases you want an OSRExit to come into existence, but you don't want to emit it right now.
    // This will emit the OSR exit in a late path. You can't be sure exactly when that will happen, but
    // you know that it will be done by the time late path emission is done. So, a linker task will
    // surely happen after that. You can use the OSRExitHandle to retrieve the exit's label.
    //
    // This API is meant to be used for things like exception handling, where some patchpoint wants to
    // have a place to jump to for OSR exit. It doesn't care where that OSR exit is emitted so long as it
    // eventually gets access to its label.
    Ref<OSRExitHandle> emitOSRExitLater(
        State&, ExitKind, const DFG::NodeOrigin&, const B3::StackmapGenerationParams&,
        uint32_t dfgNodeIndex, unsigned offset);

private:
    // This is the low-level interface. It will create a handle representing the desire to emit code for
    // an OSR exit. You can call OSRExitHandle::emitExitThunk() once you have a place to emit it. Note
    // that the above two APIs are written in terms of this and OSRExitHandle::emitExitThunk().
    Ref<OSRExitHandle> prepareOSRExitHandle(
        State&, ExitKind, const DFG::NodeOrigin&, const B3::StackmapGenerationParams&,
        uint32_t dfgNodeIndex, unsigned offset);
};

struct OSRExit : public DFG::OSRExitBase {
    OSRExit(OSRExitDescriptor*, ExitKind, CodeOrigin, CodeOrigin codeOriginForExitProfile, bool wasHoisted, uint32_t dfgNodeIndex, FixedVector<B3::ValueRep>&&);

    OSRExitDescriptor* m_descriptor;
    MacroAssemblerCodeRef<OSRExitPtrTag> m_code;
    // This tells us where to place a jump.
    CodeLocationJump<JSInternalPtrTag> m_patchableJump;
    FixedVector<B3::ValueRep> m_valueReps;

    CodeLocationJump<JSInternalPtrTag> codeLocationForRepatch(CodeBlock* ftlCodeBlock) const;
    void considerAddingAsFrequentExitSite(CodeBlock* profiledCodeBlock)
    {
        OSRExitBase::considerAddingAsFrequentExitSite(profiledCodeBlock, ExitFromFTL);
    }
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
