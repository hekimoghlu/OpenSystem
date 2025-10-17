/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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

#if ENABLE(FTL_JIT)

#include "DFGBasicBlock.h"
#include "FTLAbbreviatedTypes.h"

namespace JSC { namespace FTL {

// Represents the act of having lowered a DFG::Node to an LValue, and records the
// DFG::BasicBlock that did the lowering. The LValue is what we use most often, but
// we need to verify that we're in a block that is dominated by the one that did
// the lowering. We're guaranteed that for each DFG::Node, there will be one
// LoweredNodeValue that always dominates all uses of the DFG::Node; but there may
// be others that don't dominate and we're effectively doing opportunistic GVN on
// the lowering code.

class LoweredNodeValue {
public:
    LoweredNodeValue()
        : m_value(nullptr)
        , m_block(nullptr)
    {
    }
    
    LoweredNodeValue(LValue value, DFG::BasicBlock* block)
        : m_value(value)
        , m_block(block)
    {
        ASSERT(m_value);
        ASSERT(m_block);
    }
    
    bool isSet() const { return !!m_value; }
    bool operator!() const { return !isSet(); }
    
    LValue value() const { return m_value; }
    DFG::BasicBlock* block() const { return m_block; }
    
private:
    LValue m_value;
    DFG::BasicBlock* m_block;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
