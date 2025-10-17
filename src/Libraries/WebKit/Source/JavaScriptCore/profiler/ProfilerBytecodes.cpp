/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#include "ProfilerBytecodes.h"

#include "CodeBlock.h"
#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "ProfilerDumper.h"
#include <wtf/StringPrintStream.h>
#include <wtf/text/MakeString.h>
#include <wtf/unicode/CharacterNames.h>

namespace JSC { namespace Profiler {

Bytecodes::Bytecodes(size_t id, CodeBlock* codeBlock)
    : BytecodeSequence(codeBlock)
    , m_id(id)
    , m_inferredName(codeBlock->inferredName())
    , m_sourceCode(codeBlock->sourceCodeForTools())
    , m_hash(codeBlock->hash())
    , m_instructionCount(codeBlock->instructionsSize())
{
}

Bytecodes::~Bytecodes() = default;

void Bytecodes::dump(PrintStream& out) const
{
    out.print("#", m_hash, "(", m_id, ")");
}

Ref<JSON::Value> Bytecodes::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();

    result->setDouble(dumper.keys().m_bytecodesID, m_id);
    result->setString(dumper.keys().m_inferredName, String::fromUTF8(m_inferredName.span()));
    String sourceCode = String::fromUTF8(m_sourceCode.span());
    if (Options::abbreviateSourceCodeForProfiler()) {
        unsigned size = Options::abbreviateSourceCodeForProfiler();
        if (sourceCode.length() > size)
            sourceCode = makeString(StringView(sourceCode).left(size - 1), horizontalEllipsis);
    }
    result->setString(dumper.keys().m_sourceCode, WTFMove(sourceCode));
    result->setString(dumper.keys().m_hash, String::fromUTF8(toCString(m_hash).span()));
    result->setDouble(dumper.keys().m_instructionCount, m_instructionCount);
    addSequenceProperties(dumper, result.get());

    return result;
}

} } // namespace JSC::Profiler

