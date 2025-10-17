/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
#include "ScriptCallFrame.h"

namespace Inspector {

ScriptCallFrame::ScriptCallFrame(const String& functionName, const String& scriptName, JSC::SourceID sourceID, JSC::LineColumn lineColumn)
    : m_functionName(functionName)
    , m_scriptName(scriptName)
    , m_sourceID(sourceID)
    , m_lineColumn(lineColumn)
{
}

ScriptCallFrame::ScriptCallFrame(const String& functionName, const String& scriptName, const String& preRedirectURL, JSC::SourceID sourceID, JSC::LineColumn lineColumn)
    : m_functionName(functionName)
    , m_scriptName(scriptName)
    , m_preRedirectURL(preRedirectURL)
    , m_sourceID(sourceID)
    , m_lineColumn(lineColumn)
{
}

ScriptCallFrame::~ScriptCallFrame() = default;

bool ScriptCallFrame::isEqual(const ScriptCallFrame& o) const
{
    // Ignore sourceID in isEqual in case of identical scripts executed multiple times
    // that would get different script identifiers, but are otherwise the same.
    return m_functionName == o.m_functionName
        && m_scriptName == o.m_scriptName
        && m_preRedirectURL == o.m_preRedirectURL
        && m_lineColumn == o.m_lineColumn;
}

bool ScriptCallFrame::isNative() const
{
    return m_scriptName == "[native code]"_s;
}

Ref<Protocol::Console::CallFrame> ScriptCallFrame::buildInspectorObject() const
{
    return Protocol::Console::CallFrame::create()
        .setFunctionName(m_functionName)
        .setUrl(m_scriptName)
        .setScriptId(String::number(m_sourceID))
        .setLineNumber(m_lineColumn.line)
        .setColumnNumber(m_lineColumn.column)
        .release();
}

} // namespace Inspector
