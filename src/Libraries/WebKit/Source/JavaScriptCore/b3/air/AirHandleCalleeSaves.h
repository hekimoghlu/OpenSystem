/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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

namespace JSC {

class RegisterSetBuilder;

namespace B3 { namespace Air {

class Code;

// This utility identifies callee-save registers and tells Code. It's called from phases that
// do stack allocation. We don't do it at the end of register allocation because the real end
// of register allocation is just before stack allocation.

// FIXME: It would be cool to make this more interactive with the Air client and also more
// powerful.
// We should have shrink wrapping: https://bugs.webkit.org/show_bug.cgi?id=150458
// We should make this interact with the client: https://bugs.webkit.org/show_bug.cgi?id=150459

void handleCalleeSaves(Code&);
void handleCalleeSaves(Code&, RegisterSetBuilder);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
