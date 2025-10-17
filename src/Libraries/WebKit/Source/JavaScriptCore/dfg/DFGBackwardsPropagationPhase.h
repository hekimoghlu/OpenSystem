/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

// Infer basic information about how nodes are likely to be used by doing a block-local
// backwards flow analysis.


// Infer information after fixup has run. This should only pessimize the existing information.
// By this point, we ensure that any new uses inserted by fixup are accounted for.
//
// For example, consider:
//     b = a + 0.1
// If {b} is PureInt, then we would propagate that to {a}. But we don't actually know
// until after fixup if {a} may be used as a double.
bool performBackwardsPropagation(Graph&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
