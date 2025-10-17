/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#include "ProfilerOriginStack.h"

#include "CodeOrigin.h"
#include "InlineCallFrame.h"
#include "JSCInlines.h"
#include "ProfilerDatabase.h"

namespace JSC { namespace Profiler {

OriginStack::OriginStack(WTF::HashTableDeletedValueType)
{
    m_stack.append(Origin(WTF::HashTableDeletedValue));
}

OriginStack::OriginStack(const Origin& origin)
{
    m_stack.append(origin);
}

OriginStack::OriginStack(Database& database, CodeBlock* codeBlock, const CodeOrigin& codeOrigin)
{
    Vector<CodeOrigin> stack = codeOrigin.inlineStack();
    
    append(Origin(database, codeBlock, stack[0].bytecodeIndex()));
    
    for (unsigned i = 1; i < stack.size(); ++i) {
        append(Origin(
            database.ensureBytecodesFor(stack[i].inlineCallFrame()->baselineCodeBlock.get()),
            stack[i].bytecodeIndex()));
    }
}

OriginStack::~OriginStack() = default;

void OriginStack::append(const Origin& origin)
{
    m_stack.append(origin);
}

bool OriginStack::operator==(const OriginStack& other) const
{
    if (m_stack.size() != other.m_stack.size())
        return false;
    
    for (unsigned i = m_stack.size(); i--;) {
        if (m_stack[i] != other.m_stack[i])
            return false;
    }
    
    return true;
}

unsigned OriginStack::hash() const
{
    unsigned result = m_stack.size();
    
    for (unsigned i = m_stack.size(); i--;) {
        result *= 3;
        result += m_stack[i].hash();
    }
    
    return result;
}

void OriginStack::dump(PrintStream& out) const
{
    for (unsigned i = 0; i < m_stack.size(); ++i) {
        if (i)
            out.print(" --> ");
        out.print(m_stack[i]);
    }
}

Ref<JSON::Value> OriginStack::toJSON(Dumper& dumper) const
{
    auto result = JSON::Array::create();
    for (unsigned i = 0; i < m_stack.size(); ++i)
        result->pushValue(m_stack[i].toJSON(dumper));
    return result;
}

} } // namespace JSC::Profiler

