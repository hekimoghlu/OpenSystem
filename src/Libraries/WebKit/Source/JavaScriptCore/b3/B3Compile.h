/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

#include "B3Common.h"
#include "JITCompilation.h"

namespace JSC {

class VM;

namespace B3 {

class Procedure;

// This is a fool-proof API for compiling a Procedure to code and then running that code. You compile
// a Procedure using this API by doing:
//
// Compilation compilation = B3::compile(vm, proc);
//
// Then you keep the Compilation object alive for as long as you want to be able to run the code.
// If this API feels too high-level, you can use B3::generate() directly.

JS_EXPORT_PRIVATE Compilation compile(Procedure&);

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
