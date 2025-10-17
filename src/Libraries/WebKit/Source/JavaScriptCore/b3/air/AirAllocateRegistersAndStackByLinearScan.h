/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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

#include "CPU.h"
#include "Options.h"

namespace JSC { namespace B3 { namespace Air {

class Code;

// This implements the Poletto and Sarkar register allocator called "linear scan":
// http://dl.acm.org/citation.cfm?id=330250
//
// This is not Air's primary register allocator. We use it only when running at optLevel<2.
// That's not the default level. This register allocator is optimized primarily for running
// quickly. It's expected that improvements to this register allocator should focus on improving
// its execution time without much regard for the quality of generated code. If you want good
// code, use graph coloring.
//
// For Air's primary register allocator, see AirAllocateRegistersByGraphColoring.h|cpp.
//
// This also does stack allocation as an afterthought. It does not do any spill coalescing.
void allocateRegistersAndStackByLinearScan(Code&);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
