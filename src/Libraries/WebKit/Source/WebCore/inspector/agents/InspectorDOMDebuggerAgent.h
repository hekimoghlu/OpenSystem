/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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

#include "InspectorWebAgentBase.h"
#include <JavaScriptCore/Breakpoint.h>
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorDebuggerAgent.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/JSONValues.h>
#include <wtf/RefPtr.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class InjectedScriptManager;
}

namespace WebCore {

class Event;
class RegisteredEventListener;
class ResourceRequest;
class ScriptExecutionContext;

class InspectorDOMDebuggerAgent : public InspectorAgentBase, public Inspector::DOMDebuggerBackendDispatcherHandler, public Inspector::InspectorDebuggerAgent::Listener {
    WTF_MAKE_NONCOPYABLE(InspectorDOMDebuggerAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorDOMDebuggerAgent);
public:
    ~InspectorDOMDebuggerAgent() override;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*) override;
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason) override;
    void discardAgent() override;
    virtual bool enabled() const;

    // DOMDebuggerBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> setURLBreakpoint(const String& url, std::optional<bool>&& isRegex, RefPtr<JSON::Object>&& options) final;
    Inspector::Protocol::ErrorStringOr<void> removeURLBreakpoint(const String& url, std::optional<bool>&& isRegex) final;
    Inspector::Protocol::ErrorStringOr<void> setEventBreakpoint(Inspector::Protocol::DOMDebugger::EventBreakpointType, const String& eventName, std::optional<bool>&& caseSensitive, std::optional<bool>&& isRegex, RefPtr<JSON::Object>&& options) final;
    Inspector::Protocol::ErrorStringOr<void> removeEventBreakpoint(Inspector::Protocol::DOMDebugger::EventBreakpointType, const String& eventName, std::optional<bool>&& caseSensitive, std::optional<bool>&& isRegex) final;

    // InspectorDebuggerAgent::Listener
    void debuggerWasEnabled() override;
    void debuggerWasDisabled() override;

    // InspectorInstrumentation
    virtual void mainFrameNavigated();
    void willSendXMLHttpRequest(const String& url);
    void willFetch(const String& url);
    void willHandleEvent(ScriptExecutionContext&, Event&, const RegisteredEventListener&);
    void didHandleEvent(ScriptExecutionContext&, Event&, const RegisteredEventListener&);
    void willFireTimer(bool oneShot);
    void didFireTimer(bool oneShot);
    void willSendRequest(ResourceRequest&);
    void willSendRequestOfType(ResourceRequest&);

protected:
    InspectorDOMDebuggerAgent(WebAgentContext&, Inspector::InspectorDebuggerAgent*);
    virtual void enable();
    virtual void disable();

    virtual bool setAnimationFrameBreakpoint(Inspector::Protocol::ErrorString&, RefPtr<JSC::Breakpoint>&&) = 0;

    Inspector::InspectorDebuggerAgent* m_debuggerAgent { nullptr };

private:
    void breakOnURLIfNeeded(const String&);

    RefPtr<Inspector::DOMDebuggerBackendDispatcher> m_backendDispatcher;
    Inspector::InjectedScriptManager& m_injectedScriptManager;

    struct EventBreakpoint {
        String eventName;
        bool caseSensitive { true };
        bool isRegex { false };

        // This is only used for the breakpoint configuration (i.e. it's irrelevant when comparing).
        RefPtr<JSC::Breakpoint> specialBreakpoint;

        inline bool operator==(const EventBreakpoint& other) const
        {
            return eventName == other.eventName
                && caseSensitive == other.caseSensitive
                && isRegex == other.isRegex;
        }

        bool matches(const String&);

    private:
        std::optional<Inspector::ContentSearchUtilities::Searcher> m_eventNameSearcher;

        // Avoid having to (re)match the searcher each time an event is dispatched.
        HashSet<String> m_knownMatchingEventNames;
    };
    Vector<EventBreakpoint> m_listenerBreakpoints;
    RefPtr<JSC::Breakpoint> m_pauseOnAllIntervalsBreakpoint;
    RefPtr<JSC::Breakpoint> m_pauseOnAllListenersBreakpoint;
    RefPtr<JSC::Breakpoint> m_pauseOnAllTimeoutsBreakpoint;

    MemoryCompactRobinHoodHashMap<String, Ref<JSC::Breakpoint>> m_urlTextBreakpoints;
    MemoryCompactRobinHoodHashMap<String, Ref<JSC::Breakpoint>> m_urlRegexBreakpoints;
    RefPtr<JSC::Breakpoint> m_pauseOnAllURLsBreakpoint;
};

} // namespace WebCore
