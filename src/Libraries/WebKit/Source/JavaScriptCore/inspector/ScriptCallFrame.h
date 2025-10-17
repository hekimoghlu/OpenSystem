/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#include "DebuggerPrimitives.h"
#include "InspectorProtocolObjects.h"
#include "LineColumn.h"
#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

class ScriptCallFrame  {
public:
    using LineColumn = JSC::LineColumn;

    ScriptCallFrame(const String& functionName, const String& scriptName, JSC::SourceID, LineColumn);
    ScriptCallFrame(const String& functionName, const String& scriptName, const String& preRedirectURL, JSC::SourceID, LineColumn);
    JS_EXPORT_PRIVATE ~ScriptCallFrame();

    const String& functionName() const { return m_functionName; }
    const String& sourceURL() const { return m_scriptName; }
    const String& preRedirectURL() const { return m_preRedirectURL; }
    unsigned lineNumber() const { return m_lineColumn.line; }
    unsigned columnNumber() const { return m_lineColumn.column; }
    JSC::SourceID sourceID() const { return m_sourceID; }

    JS_EXPORT_PRIVATE bool isEqual(const ScriptCallFrame&) const;
    bool isNative() const;

    bool operator==(const ScriptCallFrame& other) const { return isEqual(other); }

    Ref<Protocol::Console::CallFrame> buildInspectorObject() const;

private:
    String m_functionName;
    String m_scriptName;
    String m_preRedirectURL;
    JSC::SourceID m_sourceID;
    LineColumn m_lineColumn;
};

} // namespace Inspector
