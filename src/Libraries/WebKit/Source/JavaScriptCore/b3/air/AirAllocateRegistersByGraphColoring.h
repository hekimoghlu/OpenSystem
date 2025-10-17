/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

// We have two register allocators, both fundamentally derived from Chaitin's Yorktown
// allocator:
// http://cs.gmu.edu/~white/CS640/p98-chaitin.pdf
//
// We have an implementation of Briggs's optimistic allocator which is derivative of Chaitin's allocator:
// http://www.cs.utexas.edu/users/mckinley/380C/lecs/briggs-thesis-1992.pdf
//
// And an implementation of Andrew Appel's Iterated Register Coalescing which is derivative of Briggs's allocator.
// http://www.cs.cmu.edu/afs/cs/academic/class/15745-s07/www/papers/george.pdf
void allocateRegistersByGraphColoring(Code&);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
