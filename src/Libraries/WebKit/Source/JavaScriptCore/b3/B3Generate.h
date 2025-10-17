/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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

namespace JSC {

class CCallHelpers;

namespace B3 {

class Procedure;
namespace Air { class Code; }

// This takes a B3::Procedure, optimizes it in-place, lowers it to Air, and prepares the Air for
// generation.
JS_EXPORT_PRIVATE void prepareForGeneration(Procedure&);

// This takes a B3::Procedure that has been prepared for generation (i.e. it has been lowered to Air and
// the Air has been prepared for generation) and generates it. This is the equivalent of calling
// Air::generate() on the Procedure::code().
JS_EXPORT_PRIVATE void generate(Procedure&, CCallHelpers&);

// This takes a B3::Procedure, optimizes it in-place, and lowers it to Air. You can then generate
// the Air to machine code using Air::prepareForGeneration() and Air::generate() on the Procedure's
// code().
void generateToAir(Procedure&);

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
