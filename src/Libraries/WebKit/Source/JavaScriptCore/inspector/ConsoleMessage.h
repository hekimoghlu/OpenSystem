/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

#include "ConsoleTypes.h"
#include "Strong.h"
#include <wtf/Forward.h>
#include <wtf/Logger.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
}

using JSC::MessageType;

namespace Inspector {

class ConsoleFrontendDispatcher;
class InjectedScriptManager;
class ScriptArguments;
class ScriptCallStack;

class ConsoleMessage {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ConsoleMessage, JS_EXPORT_PRIVATE);
    WTF_MAKE_NONCOPYABLE(ConsoleMessage);
public:
    JS_EXPORT_PRIVATE ConsoleMessage(MessageSource, MessageType, MessageLevel, const String& message, unsigned long requestIdentifier = 0, WallTime timestamp = { });
    JS_EXPORT_PRIVATE ConsoleMessage(MessageSource, MessageType, MessageLevel, const String& message, const String& url, unsigned line, unsigned column, JSC::JSGlobalObject* = nullptr, unsigned long requestIdentifier = 0, WallTime timestamp = { });
    JS_EXPORT_PRIVATE ConsoleMessage(MessageSource, MessageType, MessageLevel, const String& message, Ref<ScriptCallStack>&&, unsigned long requestIdentifier = 0, WallTime timestamp = { });
    ConsoleMessage(MessageSource, MessageType, MessageLevel, const String& message, Ref<ScriptArguments>&&, Ref<ScriptCallStack>&&, unsigned long requestIdentifier = 0, WallTime timestamp = { });
    JS_EXPORT_PRIVATE ConsoleMessage(MessageSource, MessageType, MessageLevel, const String& message, Ref<ScriptArguments>&&, JSC::JSGlobalObject* = nullptr, unsigned long requestIdentifier = 0, WallTime timestamp = { });
    JS_EXPORT_PRIVATE ConsoleMessage(MessageSource, MessageType, MessageLevel, Vector<JSONLogValue>&&, JSC::JSGlobalObject*, unsigned long requestIdentifier = 0, WallTime timestamp = { });
    JS_EXPORT_PRIVATE ~ConsoleMessage();

    void addToFrontend(ConsoleFrontendDispatcher&, InjectedScriptManager&, bool generatePreview);
    void updateRepeatCountInConsole(ConsoleFrontendDispatcher&);

    MessageSource source() const { return m_source; }
    MessageType type() const { return m_type; }
    MessageLevel level() const { return m_level; }
    const String& message() const { return m_message; }
    const String& url() const { return m_url; }
    unsigned line() const { return m_line; }
    unsigned column() const { return m_column; }
    WallTime timestamp() const { return m_timestamp; }

    JS_EXPORT_PRIVATE JSC::JSGlobalObject* globalObject() const;

    void incrementCount() { ++m_repeatCount; }

    const RefPtr<ScriptArguments>& arguments() const { return m_arguments; }
    unsigned argumentCount() const;

    bool isEqual(ConsoleMessage* msg) const;

    JS_EXPORT_PRIVATE void clear();

    JS_EXPORT_PRIVATE String toString() const;

private:
    void autogenerateMetadata(JSC::JSGlobalObject* = nullptr);

    MessageSource m_source;
    MessageType m_type;
    MessageLevel m_level;
    String m_message;
    RefPtr<ScriptArguments> m_arguments;
    RefPtr<ScriptCallStack> m_callStack;
    Vector<JSONLogValue> m_jsonLogValues;
    String m_url;
    JSC::Strong<JSC::JSGlobalObject> m_globalObject;
    unsigned m_line { 0 };
    unsigned m_column { 0 };
    unsigned m_repeatCount { 1 };
    String m_requestId;
    WallTime m_timestamp;
};

} // namespace Inspector
