/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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

// This isn't a phase - it's meant to be a utility that other phases use. Air reasons about liveness by
// reasoning about interference at boundaries between instructions. This is convenient because it works
// great in the most common case: early uses and late defs. However, this can go wrong - for example, a
// late use in one instruction doesn't actually interfere with an early def of the next instruction, but
// Air thinks that it does. It can also go wrong by having liveness incorrectly report that something is
// dead when it isn't.
//
// See https://bugs.webkit.org/show_bug.cgi?id=163548#c2 for more info.

void padInterference(Code&);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

