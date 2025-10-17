/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

class Graph;

// Convert ThreadedCPS form into SSA form. This results in a form that has:
//
// - Minimal Phi's. We use the Cytron et al (TOPLAS'91) algorithm for
//   Phi insertion. Most of the algorithm is implemented in SSACalculator
//   and Dominators.
//
// - No uses of GetLocal/SetLocal except for captured variables and flushes.
//   After this, any remaining SetLocal means Flush. PhantomLocals become
//   Phantoms. Nodes may have children that are in another basic block.
//
// - MovHints are used for OSR information, and are themselves minimal.
//   A MovHint will occur at some point after the assigning, and at Phi
//   points.
//
// - Unlike conventional SSA in which Phi functions refer to predecessors
//   and values, our SSA uses Upsilon functions to indicate values in
//   predecessors. A merge will look like:
//
//   labelA:
//       a: Thingy(...)
//       b: Upsilon(^e, @a)
//       Jump(labelC)
//
//   labelB:
//       c: OtherThingy(...)
//       d: Upsilon(^e, @c)
//       Jump(labelC)
//
//   labelC:
//       e: Phi()
//
//   Note that the Phi has no children, but the predecessors have Upsilons
//   that have a weak reference to the Phi (^e instead of @e; we store it
//   in the OpInfo rather than the AdjacencyList). Think of the Upsilon
//   as "assigning" to the "variable" associated with the Phi, and that
//   this is the one place in SSA form where you can have multiple
//   assignments.
//
//   This implies some other loosenings of SSA. For example, an Upsilon
//   may precede a Phi in the same basic block; this may arise after CFG
//   simplification. Although it's profitable for CFG simplification (or
//   some other phase) to remove these, it's not strictly necessary. As
//   well, this form allows the Upsilon to be in any block that dominates
//   the predecessor block of the Phi, which allows for block splitting to
//   ignore the possibility of introducing an extra edge between the Phi
//   and the predecessor (though normal SSA would allow this, also, with
//   the caveat that the Phi predecessor block lists would have to be
//   updated).
//
//   Fun fact: Upsilon is so named because it comes before Phi in the
//   alphabet. It can be written as "Y".

bool performSSAConversion(Graph&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
