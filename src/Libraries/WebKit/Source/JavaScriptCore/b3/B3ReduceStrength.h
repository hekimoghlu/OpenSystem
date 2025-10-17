/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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

namespace JSC { namespace B3 {

class Procedure;

// Does strength reduction, constant folding, canonicalization, CFG simplification, DCE, and very
// simple CSE. This phase runs those optimizations to fixpoint. The goal of the phase is to
// dramatically reduce the complexity of the code. In the future, it's preferable to add optimizations
// to this phase rather than creating new optimizations because then the optimizations can participate
// in the fixpoint. However, because of the many interlocking optimizations, it can be difficult to
// add sophisticated optimizations to it. For that reason we have full CSE in a different phase, for
// example.

JS_EXPORT_PRIVATE bool reduceStrength(Procedure&);

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
