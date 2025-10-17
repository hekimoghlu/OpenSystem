/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

#include "FPRInfo.h"
#include "GPRInfo.h"
#include "Reg.h"

namespace JSC {

class AssemblyHelpers;

namespace FTL {

size_t requiredScratchMemorySizeInBytes();

size_t offsetOfReg(Reg);
size_t offsetOfGPR(GPRReg);
size_t offsetOfFPR(FPRReg);

// Assumes that top-of-stack can be used as a pointer-sized scratchpad. Saves all of
// the registers into the scratch buffer such that RegisterID * sizeof(int64_t) is the
// offset of every register.
void saveAllRegisters(AssemblyHelpers& jit, char* scratchMemory);

void restoreAllRegisters(AssemblyHelpers& jit, char* scratchMemory);

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
