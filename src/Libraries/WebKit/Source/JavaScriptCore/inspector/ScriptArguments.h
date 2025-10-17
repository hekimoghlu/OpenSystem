/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include "Strong.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/MakeString.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
}

namespace Inspector {

class ScriptArguments : public RefCounted<ScriptArguments> {
public:
    JS_EXPORT_PRIVATE static Ref<ScriptArguments> create(JSC::JSGlobalObject*, Vector<JSC::Strong<JSC::Unknown>>&& arguments);
    JS_EXPORT_PRIVATE ~ScriptArguments();

    JS_EXPORT_PRIVATE JSC::JSValue argumentAt(size_t) const;
    size_t argumentCount() const { return m_arguments.size(); }

    JS_EXPORT_PRIVATE JSC::JSGlobalObject* globalObject() const;

    JS_EXPORT_PRIVATE bool getFirstArgumentAsString(String& result) const;
    JS_EXPORT_PRIVATE Vector<String> getArgumentsAsStrings() const;
    bool isEqual(const ScriptArguments&) const;

    static String truncateStringForConsoleMessage(const String& message)
    {
        constexpr size_t maxMessageLength = 10000;
        if (message.length() <= maxMessageLength)
            return message;
        return makeString(StringView(message).left(maxMessageLength), "..."_s);
    }

private:
    ScriptArguments(JSC::JSGlobalObject*, Vector<JSC::Strong<JSC::Unknown>>&& arguments);
    std::optional<String> getArgumentAtIndexAsString(size_t) const;

    JSC::Strong<JSC::JSGlobalObject> m_globalObject;
    Vector<JSC::Strong<JSC::Unknown>> m_arguments;
};

} // namespace Inspector
