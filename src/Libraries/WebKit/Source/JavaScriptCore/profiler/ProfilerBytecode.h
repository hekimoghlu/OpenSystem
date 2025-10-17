/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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

#include "JSCJSValue.h"
#include <wtf/JSONValues.h>
#include <wtf/text/CString.h>

namespace JSC {

enum OpcodeID : unsigned;

namespace Profiler {

class Dumper;

class Bytecode {
public:
    Bytecode()
        : m_bytecodeIndex(std::numeric_limits<unsigned>::max())
    {
    }
    
    Bytecode(unsigned bytecodeIndex, OpcodeID opcodeID, const CString& description)
        : m_bytecodeIndex(bytecodeIndex)
        , m_opcodeID(opcodeID)
        , m_description(description)
    {
    }
    
    unsigned bytecodeIndex() const { return m_bytecodeIndex; }
    OpcodeID opcodeID() const { return m_opcodeID; }
    const CString& description() const { return m_description; }
    
    Ref<JSON::Value> toJSON(Dumper&) const;
private:
    unsigned m_bytecodeIndex;
    OpcodeID m_opcodeID;
    CString m_description;
};

inline unsigned getBytecodeIndexForBytecode(Bytecode* bytecode) { return bytecode->bytecodeIndex(); }

} } // namespace JSC::Profiler
