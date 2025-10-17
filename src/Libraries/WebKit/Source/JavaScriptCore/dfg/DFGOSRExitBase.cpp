/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include "DFGOSRExitBase.h"

#if ENABLE(DFG_JIT)

#include "InlineCallFrame.h"

namespace JSC { namespace DFG {

void OSRExitBase::considerAddingAsFrequentExitSiteSlow(CodeBlock* profiledCodeBlock, ExitingJITType jitType)
{
    CodeBlock* sourceProfiledCodeBlock =
        baselineCodeBlockForOriginAndBaselineCodeBlock(
            m_codeOriginForExitProfile, profiledCodeBlock);
    if (sourceProfiledCodeBlock) {
        ExitingInlineKind inlineKind;
        if (m_codeOriginForExitProfile.inlineCallFrame())
            inlineKind = ExitFromInlined;
        else
            inlineKind = ExitFromNotInlined;
        
        FrequentExitSite site;
        if (m_wasHoisted)
            site = FrequentExitSite(HoistingFailed, jitType, inlineKind);
        else
            site = FrequentExitSite(m_codeOriginForExitProfile.bytecodeIndex(), m_kind, jitType, inlineKind);
        ExitProfile::add(sourceProfiledCodeBlock, site);
    }
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

