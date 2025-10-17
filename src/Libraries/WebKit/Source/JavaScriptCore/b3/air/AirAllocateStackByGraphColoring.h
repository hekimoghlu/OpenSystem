/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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

namespace JSC { namespace B3 { namespace Air {

class Code;

// This allocates StackSlots to places on the stack. It first allocates the pinned ones in index
// order and then it allocates the rest using first fit. Takes the opportunity to kill dead
// assignments to stack slots, since it knows which ones are live. Also coalesces spill slots
// whenever possible.
//
// This is meant to be an optimal stack allocator, focused on generating great code. It's not
// particularly fast, though.

void allocateStackByGraphColoring(Code&);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
