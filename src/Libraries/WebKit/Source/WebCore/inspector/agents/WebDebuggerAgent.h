/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include <JavaScriptCore/InspectorDebuggerAgent.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class EventListener;
class EventTarget;
class InstrumentingAgents;
class RegisteredEventListener;
class TimerBase;

class WebDebuggerAgent : public Inspector::InspectorDebuggerAgent {
    WTF_MAKE_NONCOPYABLE(WebDebuggerAgent);
    WTF_MAKE_TZONE_ALLOCATED(WebDebuggerAgent);
public:
    ~WebDebuggerAgent() override;
    bool enabled() const override;

    // InspectorInstrumentation
    void didAddEventListener(EventTarget&, const AtomString& eventType, EventListener&, bool capture);
    void willRemoveEventListener(EventTarget&, const AtomString& eventType, EventListener&, bool capture);
    void willHandleEvent(const RegisteredEventListener&);
    void didHandleEvent(const RegisteredEventListener&);
    int willPostMessage();
    void didPostMessage(int postMessageIdentifier, JSC::JSGlobalObject&);
    void didFailPostMessage(int postMessageIdentifier);
    void willDispatchPostMessage(int postMessageIdentifier);
    void didDispatchPostMessage(int postMessageIdentifier);

protected:
    WebDebuggerAgent(WebAgentContext&);
    void internalEnable() override;
    void internalDisable(bool isBeingDestroyed) override;

    void didClearAsyncStackTraceData() final;

    InstrumentingAgents& m_instrumentingAgents;

private:
    HashMap<const RegisteredEventListener*, int> m_registeredEventListeners;
    HashMap<const RegisteredEventListener*, int> m_dispatchedEventListeners;
    HashSet<int> m_postMessageTasks;
    int m_nextEventListenerIdentifier { 1 };
    int m_nextPostMessageIdentifier { 1 };
};

} // namespace WebCore
