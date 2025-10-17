/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

class BasicBlock;
class BlockInsertionSet;
class Graph;

// Inserts dummy basic blocks before any loop headers that don't already have
// a single non-loop predecessor.

bool performLoopPreHeaderCreation(Graph&);

// Creates a new basic block (a pre-header) that jumps to the given block. All
// predecessors of the given block that aren't dominated by it are rerouted to
// the pre-header.
//
// This function is used internally and it's used in a surgical fashion by
// OSREntrypointCreationPhase.
//
// Note that executing this function requires having an intact dominators
// analysis. You should run that analysis before doing damage to the CFG, even
// if said damage is done before you call this.
BasicBlock* createPreHeader(Graph&, BlockInsertionSet&, BasicBlock*);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
