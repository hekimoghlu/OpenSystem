/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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

#include "InspectorAgentBase.h"
#include "InspectorBackendDispatchers.h"
#include "InspectorFrontendDispatchers.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>

namespace JSC {
class CallFrame;
}

namespace Inspector {

class ConsoleMessage;
class InjectedScriptManager;
class InspectorHeapAgent;
class ScriptArguments;
class ScriptCallStack;

class JS_EXPORT_PRIVATE InspectorConsoleAgent : public InspectorAgentBase, public ConsoleBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorConsoleAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorConsoleAgent);
public:
    InspectorConsoleAgent(AgentContext&);
    ~InspectorConsoleAgent() override;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;
    void discardValues() final;

    // ConsoleBackendDispatcherHandler
    Protocol::ErrorStringOr<void> enable() final;
    Protocol::ErrorStringOr<void> disable() final;
    Protocol::ErrorStringOr<void> clearMessages() override;
    Protocol::ErrorStringOr<void> setConsoleClearAPIEnabled(bool) override;
    Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Protocol::Console::Channel>>> getLoggingChannels() override;
    Protocol::ErrorStringOr<void> setLoggingChannelLevel(Protocol::Console::ChannelSource, Protocol::Console::ChannelLevel) override;

    void setHeapAgent(InspectorHeapAgent* agent) { m_heapAgent = agent; }

    bool enabled() const { return m_enabled; }
    bool developerExtrasEnabled() const;

    // InspectorInstrumentation
    void mainFrameNavigated();

    void addMessageToConsole(std::unique_ptr<ConsoleMessage>);

    void startTiming(JSC::JSGlobalObject*, const String& label);
    void logTiming(JSC::JSGlobalObject*, const String& label, Ref<ScriptArguments>&&);
    void stopTiming(JSC::JSGlobalObject*, const String& label);
    void takeHeapSnapshot(const String& title);
    void count(JSC::JSGlobalObject*, const String& label);
    void countReset(JSC::JSGlobalObject*, const String& label);

protected:
    void addConsoleMessage(std::unique_ptr<ConsoleMessage>);
    void clearMessages(Protocol::Console::ClearReason);

    InjectedScriptManager& m_injectedScriptManager;
    std::unique_ptr<ConsoleFrontendDispatcher> m_frontendDispatcher;
    RefPtr<ConsoleBackendDispatcher> m_backendDispatcher;
    InspectorHeapAgent* m_heapAgent { nullptr };

    Vector<std::unique_ptr<ConsoleMessage>> m_consoleMessages;
    int m_expiredConsoleMessageCount { 0 };
    UncheckedKeyHashMap<String, unsigned> m_counts;
    UncheckedKeyHashMap<String, MonotonicTime> m_times;
    bool m_enabled { false };
    bool m_isAddingMessageToFrontend { false };
    bool m_consoleClearAPIEnabled { true };
};

} // namespace Inspector
