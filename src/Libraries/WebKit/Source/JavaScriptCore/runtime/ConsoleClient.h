/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class ConsoleClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<JSC::ConsoleClient> : std::true_type { };
}

namespace Inspector {
class ScriptArguments;
}

namespace JSC {

class CallFrame;
class JSGlobalObject;

class ConsoleClient : public CanMakeWeakPtr<ConsoleClient> {
public:
    virtual ~ConsoleClient() { }

    JS_EXPORT_PRIVATE static void printConsoleMessage(MessageSource, MessageType, MessageLevel, const String& message, const String& url, unsigned lineNumber, unsigned columnNumber);
    JS_EXPORT_PRIVATE static void printConsoleMessageWithArguments(MessageSource, MessageType, MessageLevel, JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);

    void logWithLevel(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&, MessageLevel);
    void clear(JSGlobalObject*);
    void dir(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void dirXML(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void table(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void trace(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void assertion(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void group(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void groupCollapsed(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);
    void groupEnd(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&);

    virtual void messageWithTypeAndLevel(MessageType, MessageLevel, JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) = 0;
    virtual void count(JSGlobalObject*, const String& label) = 0;
    virtual void countReset(JSGlobalObject*, const String& label) = 0;
    virtual void profile(JSGlobalObject*, const String& title) = 0;
    virtual void profileEnd(JSGlobalObject*, const String& title) = 0;
    virtual void takeHeapSnapshot(JSGlobalObject*, const String& title) = 0;
    virtual void time(JSGlobalObject*, const String& label) = 0;
    virtual void timeLog(JSGlobalObject*, const String& label, Ref<Inspector::ScriptArguments>&&) = 0;
    virtual void timeEnd(JSGlobalObject*, const String& label) = 0;
    virtual void timeStamp(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) = 0;
    virtual void record(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) = 0;
    virtual void recordEnd(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) = 0;
    virtual void screenshot(JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) = 0;

private:
    enum ArgumentRequirement { ArgumentRequired, ArgumentNotRequired };
    void internalMessageWithTypeAndLevel(MessageType, MessageLevel, JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&, ArgumentRequirement);
};

} // namespace JSC
