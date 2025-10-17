/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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

#include "DFGBasicBlock.h"

namespace JSC { namespace DFG {

class Graph;

// Computes BasicBlock::ssa->availabilityAtHead/Tail. This is a forward flow type inference
// over MovHints and SetLocals. This analysis is run directly by the Plan for preparing for
// lowering to B3 IR, but it can also be used as a utility. Note that if you run it before
// stack layout, all of the flush availability will omit the virtual register - but it will
// tell you the format.

bool performOSRAvailabilityAnalysis(Graph&);

// Unlike the phase above this function doesn't mutate the graph's BasicBlock SSA metadata. Also, does nothing if !validationEnabled()
void validateOSRExitAvailability(Graph&);

// Local calculator for figuring out the availability at any node in a basic block. Requires
// having run the availability analysis.
class LocalOSRAvailabilityCalculator {
public:
    LocalOSRAvailabilityCalculator(Graph&);
    ~LocalOSRAvailabilityCalculator();
    
    void beginBlock(BasicBlock*);
    void endBlock(BasicBlock*); // Useful if you want to get data for the end of the block. You don't need to call this if you did beginBlock() and then executeNode() for every node.
    void executeNode(Node*);
    
    AvailabilityMap m_availability;
    Graph& m_graph;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
