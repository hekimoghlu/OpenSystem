/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

// Tells you if an operation that filters type (i.e. does a type check/speculation) will always
// exit. Formally, this means that the proven type of a value prior to the filter was not
// bottom (i.e. not "clear" or "SpecEmpty") but becomes bottom as a result of executing the
// filter.
//
// Note that per this definition, a filter will not return Contradiction if the node's proven
// type was already bottom. This is necessary because we have this yucky convention of using
// a proven type of bottom for nodes that don't hold JS values, like Phi nodes in ThreadedCPS
// and storage nodes.
enum FiltrationResult {
    // Means that this operation may not always exit.
    FiltrationOK,
    
    // Means that this operation will always exit.
    Contradiction
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
