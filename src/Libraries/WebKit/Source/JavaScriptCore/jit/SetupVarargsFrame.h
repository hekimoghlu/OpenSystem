/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

#include "CCallHelpers.h"

namespace JSC {

void emitSetVarargsFrame(CCallHelpers&, GPRReg lengthGPR, bool lengthIncludesThis, GPRReg numUsedSlotsGPR, GPRReg resultGPR);

// Assumes that SP refers to the last in-use stack location, and after this returns SP will point to
// the newly created frame plus the native header. scratchGPR2 may be the same as numUsedSlotsGPR.
void emitSetupVarargsFrameFastCase(VM&, CCallHelpers&, GPRReg numUsedSlotsGPR, GPRReg scratchGPR1, GPRReg scratchGPR2, GPRReg scratchGPR3, InlineCallFrame*, unsigned firstVarArgOffset, CCallHelpers::JumpList& slowCase);

} // namespace JSC

#endif // ENABLE(JIT)
