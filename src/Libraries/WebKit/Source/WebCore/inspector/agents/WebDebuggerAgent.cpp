/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "WebDebuggerAgent.h"

#include "EventListener.h"
#include "EventTarget.h"
#include "InstrumentingAgents.h"
#include "ScriptExecutionContext.h"
#include "Timer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebDebuggerAgent);

WebDebuggerAgent::WebDebuggerAgent(WebAgentContext& context)
    : InspectorDebuggerAgent(context)
    , m_instrumentingAgents(context.instrumentingAgents)
{
}

WebDebuggerAgent::~WebDebuggerAgent() = default;

bool WebDebuggerAgent::enabled() const
{
    return m_instrumentingAgents.enabledWebDebuggerAgent() == this && InspectorDebuggerAgent::enabled();
}

void WebDebuggerAgent::internalEnable()
{
    m_instrumentingAgents.setEnabledWebDebuggerAgent(this);

    InspectorDebuggerAgent::internalEnable();
}

void WebDebuggerAgent::internalDisable(bool isBeingDestroyed)
{
    m_instrumentingAgents.setEnabledWebDebuggerAgent(nullptr);

    InspectorDebuggerAgent::internalDisable(isBeingDestroyed);
}

void WebDebuggerAgent::didAddEventListener(EventTarget& target, const AtomString& eventType, EventListener& listener, bool capture)
{
    if (!breakpointsActive())
        return;

    auto& eventListeners = target.eventListeners(eventType);
    auto position = eventListeners.findIf([&](auto& registeredListener) {
        return &registeredListener->callback() == &listener && registeredListener->useCapture() == capture;
    });
    if (position == notFound)
        return;

    auto& registeredListener = eventListeners.at(position);
    if (m_registeredEventListeners.contains(registeredListener.get()))
        return;

    auto* globalObject = target.scriptExecutionContext()->globalObject();
    if (!globalObject)
        return;

    int identifier = m_nextEventListenerIdentifier++;
    m_registeredEventListeners.set(registeredListener.get(), identifier);

    didScheduleAsyncCall(globalObject, InspectorDebuggerAgent::AsyncCallType::EventListener, identifier, registeredListener->isOnce());
}

void WebDebuggerAgent::willRemoveEventListener(EventTarget& target, const AtomString& eventType, EventListener& listener, bool capture)
{
    auto& eventListeners = target.eventListeners(eventType);
    size_t listenerIndex = eventListeners.findIf([&](auto& registeredListener) {
        return &registeredListener->callback() == &listener && registeredListener->useCapture() == capture;
    });

    if (listenerIndex == notFound)
        return;

    int identifier = m_registeredEventListeners.take(eventListeners[listenerIndex].get());
    didCancelAsyncCall(InspectorDebuggerAgent::AsyncCallType::EventListener, identifier);
}

void WebDebuggerAgent::willHandleEvent(const RegisteredEventListener& listener)
{
    auto it = m_registeredEventListeners.find(&listener);
    if (it == m_registeredEventListeners.end())
        return;

    // Save the identifier for the listener we're about to dispatch an event to in case it's removed.
    m_dispatchedEventListeners.set(&listener, it->value);

    willDispatchAsyncCall(InspectorDebuggerAgent::AsyncCallType::EventListener, it->value);
}

void WebDebuggerAgent::didHandleEvent(const RegisteredEventListener& listener)
{
    auto it = m_dispatchedEventListeners.find(&listener);
    if (it == m_dispatchedEventListeners.end())
        return;

    didDispatchAsyncCall(InspectorDebuggerAgent::AsyncCallType::EventListener, it->value);

    m_dispatchedEventListeners.remove(it);
}

int WebDebuggerAgent::willPostMessage()
{
    if (!breakpointsActive())
        return 0;

    auto postMessageIdentifier = m_nextPostMessageIdentifier++;
    m_postMessageTasks.add(postMessageIdentifier);
    return postMessageIdentifier;
}

void WebDebuggerAgent::didPostMessage(int postMessageIdentifier, JSC::JSGlobalObject& state)
{
    if (!breakpointsActive())
        return;

    if (!postMessageIdentifier || !m_postMessageTasks.contains(postMessageIdentifier))
        return;

    didScheduleAsyncCall(&state, InspectorDebuggerAgent::AsyncCallType::PostMessage, postMessageIdentifier, true);
}

void WebDebuggerAgent::didFailPostMessage(int postMessageIdentifier)
{
    if (!postMessageIdentifier)
        return;

    auto it = m_postMessageTasks.find(postMessageIdentifier);
    if (it == m_postMessageTasks.end())
        return;

    didCancelAsyncCall(InspectorDebuggerAgent::AsyncCallType::PostMessage, postMessageIdentifier);

    m_postMessageTasks.remove(it);
}

void WebDebuggerAgent::willDispatchPostMessage(int postMessageIdentifier)
{
    if (!postMessageIdentifier || !m_postMessageTasks.contains(postMessageIdentifier))
        return;

    willDispatchAsyncCall(InspectorDebuggerAgent::AsyncCallType::PostMessage, postMessageIdentifier);
}

void WebDebuggerAgent::didDispatchPostMessage(int postMessageIdentifier)
{
    if (!postMessageIdentifier)
        return;

    auto it = m_postMessageTasks.find(postMessageIdentifier);
    if (it == m_postMessageTasks.end())
        return;

    didDispatchAsyncCall(InspectorDebuggerAgent::AsyncCallType::PostMessage, postMessageIdentifier);

    m_postMessageTasks.remove(it);
}

void WebDebuggerAgent::didClearAsyncStackTraceData()
{
    InspectorDebuggerAgent::didClearAsyncStackTraceData();

    m_registeredEventListeners.clear();
    m_dispatchedEventListeners.clear();
    m_postMessageTasks.clear();
    m_nextEventListenerIdentifier = 1;
    m_nextPostMessageIdentifier = 1;
}

} // namespace WebCore
