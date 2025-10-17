/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#include "DFGNode.h"
#include "DataFormat.h"
#include "FTLAbbreviatedTypes.h"
#include "FTLRecoveryOpcode.h"

namespace JSC { namespace FTL {

class AvailableRecovery {
public:
    AvailableRecovery()
        : m_node(nullptr)
        , m_format(DataFormatNone)
        , m_opcode(AddRecovery)
        , m_left(nullptr)
        , m_right(nullptr)
    {
    }
    
    AvailableRecovery(DFG::Node* node, RecoveryOpcode opcode, LValue left, LValue right, DataFormat format)
        : m_node(node)
        , m_format(format)
        , m_opcode(opcode)
        , m_left(left)
        , m_right(right)
    {
    }
    
    DFG::Node* node() const { return m_node; }
    DataFormat format() const { return m_format; }
    RecoveryOpcode opcode() const { return m_opcode; }
    LValue left() const { return m_left; }
    LValue right() const { return m_right; }
    
    void dump(PrintStream&) const;
    
private:
    DFG::Node* m_node;
    DataFormat m_format;
    RecoveryOpcode m_opcode;
    LValue m_left;
    LValue m_right;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
