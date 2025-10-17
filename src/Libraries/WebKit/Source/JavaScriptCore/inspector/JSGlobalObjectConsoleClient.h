/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#include "ConsoleClient.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

using JSC::MessageType;

namespace Inspector {

class InspectorConsoleAgent;
class InspectorDebuggerAgent;
class InspectorScriptProfilerAgent;

class JSGlobalObjectConsoleClient final : public JSC::ConsoleClient {
    WTF_MAKE_TZONE_ALLOCATED(JSGlobalObjectConsoleClient);
public:
    explicit JSGlobalObjectConsoleClient(InspectorConsoleAgent*);
    ~JSGlobalObjectConsoleClient() final { }

    static bool logToSystemConsole();
    static void setLogToSystemConsole(bool);

    void setDebuggerAgent(InspectorDebuggerAgent* agent) { m_debuggerAgent = agent; }
    void setPersistentScriptProfilerAgent(InspectorScriptProfilerAgent* agent) { m_scriptProfilerAgent = agent; }

private:
    void messageWithTypeAndLevel(MessageType, MessageLevel, JSC::JSGlobalObject*, Ref<ScriptArguments>&&) final;
    void count(JSC::JSGlobalObject*, const String& label) final;
    void countReset(JSC::JSGlobalObject*, const String& label) final;
    void profile(JSC::JSGlobalObject*, const String& title) final;
    void profileEnd(JSC::JSGlobalObject*, const String& title) final;
    void takeHeapSnapshot(JSC::JSGlobalObject*, const String& title) final;
    void time(JSC::JSGlobalObject*, const String& label) final;
    void timeLog(JSC::JSGlobalObject*, const String& label, Ref<ScriptArguments>&&) final;
    void timeEnd(JSC::JSGlobalObject*, const String& label) final;
    void timeStamp(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) final;
    void record(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) final;
    void recordEnd(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) final;
    void screenshot(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) final;

    void warnUnimplemented(const String& method);
    void internalAddMessage(MessageType, MessageLevel, JSC::JSGlobalObject*, Ref<ScriptArguments>&&);

    void startConsoleProfile();
    void stopConsoleProfile();

    InspectorConsoleAgent* m_consoleAgent;
    InspectorDebuggerAgent* m_debuggerAgent { nullptr };
    InspectorScriptProfilerAgent* m_scriptProfilerAgent { nullptr };
    Vector<String> m_profiles;
    bool m_profileRestoreBreakpointActiveValue { false };
};

}
