/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#include "InspectorConsoleAgent.h"

#include "ConsoleMessage.h"
#include "InjectedScriptManager.h"
#include "InspectorHeapAgent.h"
#include "ScriptArguments.h"
#include "ScriptCallStackFactory.h"
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace Inspector {

static constexpr unsigned maximumConsoleMessages = 100;
static constexpr int expireConsoleMessagesStep = 10;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorConsoleAgent);

InspectorConsoleAgent::InspectorConsoleAgent(AgentContext& context)
    : InspectorAgentBase("Console"_s)
    , m_injectedScriptManager(context.injectedScriptManager)
    , m_frontendDispatcher(makeUnique<ConsoleFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(ConsoleBackendDispatcher::create(context.backendDispatcher, this))
{
}

InspectorConsoleAgent::~InspectorConsoleAgent() = default;

void InspectorConsoleAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
}

void InspectorConsoleAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    disable();
}

void InspectorConsoleAgent::discardValues()
{
    m_consoleMessages.clear();
    m_expiredConsoleMessageCount = 0;
}

Protocol::ErrorStringOr<void> InspectorConsoleAgent::enable()
{
    if (m_enabled)
        return { };

    m_enabled = true;

    if (m_expiredConsoleMessageCount) {
        ConsoleMessage expiredMessage(MessageSource::Other, MessageType::Log, MessageLevel::Warning, makeString(m_expiredConsoleMessageCount, " console messages are not shown."_s));
        expiredMessage.addToFrontend(*m_frontendDispatcher, m_injectedScriptManager, false);
    }

    for (auto& message : m_consoleMessages)
        message->addToFrontend(*m_frontendDispatcher, m_injectedScriptManager, false);

    return { };
}

Protocol::ErrorStringOr<void> InspectorConsoleAgent::disable()
{
    if (!m_enabled)
        return { };

    m_enabled = false;

    return { };
}

Protocol::ErrorStringOr<void> InspectorConsoleAgent::clearMessages()
{
    clearMessages(Inspector::Protocol::Console::ClearReason::Frontend);
    return { };
}

Protocol::ErrorStringOr<void> InspectorConsoleAgent::setConsoleClearAPIEnabled(bool enabled)
{
    m_consoleClearAPIEnabled = enabled;
    return { };
}

bool InspectorConsoleAgent::developerExtrasEnabled() const
{
    return m_injectedScriptManager.inspectorEnvironment().developerExtrasEnabled();
}

void InspectorConsoleAgent::mainFrameNavigated()
{
    clearMessages(Inspector::Protocol::Console::ClearReason::MainFrameNavigation);

    m_times.clear();
    m_counts.clear();
}

void InspectorConsoleAgent::clearMessages(Inspector::Protocol::Console::ClearReason reason)
{
    if (!m_consoleClearAPIEnabled && reason == Inspector::Protocol::Console::ClearReason::ConsoleAPI)
        return;

    m_consoleMessages.clear();
    m_expiredConsoleMessageCount = 0;

    m_injectedScriptManager.releaseObjectGroup("console"_s);

    if (m_enabled)
        m_frontendDispatcher->messagesCleared(reason);
}

void InspectorConsoleAgent::addMessageToConsole(std::unique_ptr<ConsoleMessage> message)
{
    if (message->type() == MessageType::Clear)
        clearMessages(Inspector::Protocol::Console::ClearReason::ConsoleAPI);

    addConsoleMessage(WTFMove(message));
}

void InspectorConsoleAgent::startTiming(JSC::JSGlobalObject* globalObject, const String& label)
{
    ASSERT(!label.isNull());
    if (label.isNull())
        return;

    auto result = m_times.add(label, MonotonicTime::now());

    if (!result.isNewEntry) {
        // FIXME: Send an enum to the frontend for localization?
        auto warning = makeString("Timer \""_s, ScriptArguments::truncateStringForConsoleMessage(label), "\" already exists"_s);
        addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Timing, MessageLevel::Warning, warning, createScriptCallStackForConsole(globalObject, 1)));
    }
}

void InspectorConsoleAgent::logTiming(JSC::JSGlobalObject* globalObject, const String& label, Ref<ScriptArguments>&& arguments)
{
    ASSERT(!label.isNull());
    if (label.isNull())
        return;

    auto callStack = createScriptCallStackForConsole(globalObject, 1);

    auto it = m_times.find(label);
    if (it == m_times.end()) {
        // FIXME: Send an enum to the frontend for localization?
        auto warning = makeString("Timer \""_s, ScriptArguments::truncateStringForConsoleMessage(label), "\" does not exist"_s);
        addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Timing, MessageLevel::Warning, warning, WTFMove(callStack)));
        return;
    }

    MonotonicTime startTime = it->value;
    Seconds elapsed = MonotonicTime::now() - startTime;
    auto message = makeString(ScriptArguments::truncateStringForConsoleMessage(label), ": "_s, FormattedNumber::fixedWidth(elapsed.milliseconds(), 3), "ms"_s);
    addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Timing, MessageLevel::Debug, message, WTFMove(arguments), WTFMove(callStack)));
}

void InspectorConsoleAgent::stopTiming(JSC::JSGlobalObject* globalObject, const String& label)
{
    ASSERT(!label.isNull());
    if (label.isNull())
        return;

    auto callStack = createScriptCallStackForConsole(globalObject, 1);

    auto it = m_times.find(label);
    if (it == m_times.end()) {
        // FIXME: Send an enum to the frontend for localization?
        auto warning = makeString("Timer \""_s, ScriptArguments::truncateStringForConsoleMessage(label), "\" does not exist"_s);
        addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Timing, MessageLevel::Warning, warning, WTFMove(callStack)));
        return;
    }

    MonotonicTime startTime = it->value;
    Seconds elapsed = MonotonicTime::now() - startTime;
    auto message = makeString(ScriptArguments::truncateStringForConsoleMessage(label), ": "_s, FormattedNumber::fixedWidth(elapsed.milliseconds(), 3), "ms"_s);
    addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Timing, MessageLevel::Debug, message, WTFMove(callStack)));

    m_times.remove(it);
}

void InspectorConsoleAgent::takeHeapSnapshot(const String& title)
{
    if (!m_heapAgent)
        return;

    auto result = m_heapAgent->snapshot();
    if (!result)
        return;

    auto [timestamp, snapshotData] = WTFMove(result.value());
    m_frontendDispatcher->heapSnapshot(timestamp, snapshotData, title);
}

void InspectorConsoleAgent::count(JSC::JSGlobalObject* globalObject, const String& label)
{
    auto result = m_counts.add(label, 1);
    if (!result.isNewEntry)
        result.iterator->value += 1;

    // FIXME: Web Inspector should have a better UI for counters, but for now we just log an updated counter value.

    auto message = makeString(ScriptArguments::truncateStringForConsoleMessage(label), ": "_s, result.iterator->value);
    addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Log, MessageLevel::Debug, message, createScriptCallStackForConsole(globalObject, 1)));
}

void InspectorConsoleAgent::countReset(JSC::JSGlobalObject* globalObject, const String& label)
{
    auto it = m_counts.find(label);
    if (it == m_counts.end()) {
        // FIXME: Send an enum to the frontend for localization?
        auto warning = makeString("Counter \""_s, ScriptArguments::truncateStringForConsoleMessage(label), "\" does not exist"_s);
        addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::ConsoleAPI, MessageType::Log, MessageLevel::Warning, warning, createScriptCallStackForConsole(globalObject, 1)));
        return;
    }

    it->value = 0;

    // FIXME: Web Inspector should have a better UI for counters, but for now we just log an updated counter value.
}

void InspectorConsoleAgent::addConsoleMessage(std::unique_ptr<ConsoleMessage> consoleMessage)
{
    ASSERT_ARG(consoleMessage, consoleMessage);

    ConsoleMessage* previousMessage = m_consoleMessages.isEmpty() ? nullptr : m_consoleMessages.last().get();

    if (previousMessage && previousMessage->isEqual(consoleMessage.get())) {
        previousMessage->incrementCount();
        if (m_enabled)
            previousMessage->updateRepeatCountInConsole(*m_frontendDispatcher);
    } else {
        if (m_enabled) {
            auto generatePreview = !m_isAddingMessageToFrontend;
            SetForScope isAddingMessageToFrontend(m_isAddingMessageToFrontend, true);

            consoleMessage->addToFrontend(*m_frontendDispatcher, m_injectedScriptManager, generatePreview);
        }

        m_consoleMessages.append(WTFMove(consoleMessage));
        if (m_consoleMessages.size() >= maximumConsoleMessages) {
            m_expiredConsoleMessageCount += expireConsoleMessagesStep;
            m_consoleMessages.remove(0, expireConsoleMessagesStep);
        }
    }
}

Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Protocol::Console::Channel>>> InspectorConsoleAgent::getLoggingChannels()
{
    // Default implementation has no logging channels.
    return JSON::ArrayOf<Protocol::Console::Channel>::create();
}

Protocol::ErrorStringOr<void> InspectorConsoleAgent::setLoggingChannelLevel(Protocol::Console::ChannelSource, Protocol::Console::ChannelLevel)
{
    return makeUnexpected("Not supported"_s);
}

} // namespace Inspector
