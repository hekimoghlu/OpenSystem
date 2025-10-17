/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#include "config.h"
#include "ProfilerBytecodeSequence.h"

#include "CodeBlock.h"
#include "JSCInlines.h"
#include "ProfilerDumper.h"
#include <wtf/StringPrintStream.h>

namespace JSC { namespace Profiler {

BytecodeSequence::BytecodeSequence(CodeBlock* codeBlock)
{
    StringPrintStream out;

    {
        unsigned index = 0;
        ConcurrentJSLocker locker(codeBlock->valueProfileLock());
        for (auto& profile : codeBlock->argumentValueProfiles()) {
            CString description = profile.briefDescription(locker);
            if (!description.length())
                continue;
            out.reset();
            out.print("arg", index++, ": ", description);
            m_header.append(out.toCString());
        }
    }
    
    ICStatusMap statusMap;
    codeBlock->getICStatusMap(statusMap);
    
    for (unsigned bytecodeIndex = 0; bytecodeIndex < codeBlock->instructions().size();) {
        out.reset();
        codeBlock->dumpBytecode(out, bytecodeIndex, statusMap);
        auto instruction = codeBlock->instructions().at(bytecodeIndex);
        OpcodeID opcodeID = instruction->opcodeID();
        m_sequence.append(Bytecode(bytecodeIndex, opcodeID, out.toCString()));
        bytecodeIndex += instruction->size();
    }
}

BytecodeSequence::~BytecodeSequence() = default;

unsigned BytecodeSequence::indexForBytecodeIndex(unsigned bytecodeIndex) const
{
    return binarySearch<Bytecode, unsigned>(m_sequence, m_sequence.size(), bytecodeIndex, getBytecodeIndexForBytecode) - m_sequence.begin();
}

const Bytecode& BytecodeSequence::forBytecodeIndex(unsigned bytecodeIndex) const
{
    return at(indexForBytecodeIndex(bytecodeIndex));
}

void BytecodeSequence::addSequenceProperties(Dumper& dumper, JSON::Object& result) const
{
    Ref jsonHeader = JSON::Array::create();
    for (auto& header : m_header)
        jsonHeader->pushString(String::fromUTF8(header.span()));
    result.setValue(dumper.keys().m_header, WTFMove(jsonHeader));

    Ref jsonSequence = JSON::Array::create();
    for (auto& sequence : m_sequence)
        jsonSequence->pushValue(sequence.toJSON(dumper));
    result.setValue(dumper.keys().m_bytecode, WTFMove(jsonSequence));
}

} } // namespace JSC::Profiler

