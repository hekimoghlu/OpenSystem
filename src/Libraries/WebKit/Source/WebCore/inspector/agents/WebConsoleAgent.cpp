/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "WebConsoleAgent.h"

#include "CommandLineAPIHost.h"
#include "InspectorNetworkAgent.h"
#include "InspectorWebAgentBase.h"
#include "JSExecState.h"
#include "LocalDOMWindow.h"
#include "Logging.h"
#include "ResourceError.h"
#include "ResourceResponse.h"
#include "WebInjectedScriptManager.h"
#include <JavaScriptCore/ConsoleMessage.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/ScriptArguments.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

using namespace Inspector;

static String blockedTrackerErrorMessage(const ResourceError& error)
{
#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
    if (error.blockedKnownTracker())
        return "Blocked connection to known tracker"_s;
#else
    UNUSED_PARAM(error);
#endif
    return { };
}

WebConsoleAgent::WebConsoleAgent(WebAgentContext& context)
    : InspectorConsoleAgent(context)
{
}

void WebConsoleAgent::frameWindowDiscarded(LocalDOMWindow& window)
{
    if (auto* document = window.document()) {
        for (auto& message : m_consoleMessages) {
            if (executionContext(message->globalObject()) == document)
                message->clear();
        }
    }
    static_cast<WebInjectedScriptManager&>(m_injectedScriptManager).discardInjectedScriptsFor(window);
}

void WebConsoleAgent::didReceiveResponse(ResourceLoaderIdentifier requestIdentifier, const ResourceResponse& response)
{
    if (response.httpStatusCode() >= 400) {
        auto message = makeString("Failed to load resource: the server responded with a status of "_s, response.httpStatusCode(), " ("_s, ScriptArguments::truncateStringForConsoleMessage(response.httpStatusText()), ')');
        addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::Network, MessageType::Log, MessageLevel::Error, message, response.url().string(), 0, 0, nullptr, requestIdentifier.toUInt64()));
    }
}

void WebConsoleAgent::didFailLoading(ResourceLoaderIdentifier requestIdentifier, const ResourceError& error)
{
    if (error.domain() == InspectorNetworkAgent::errorDomain())
        return;

    // Report failures only.
    if (error.isCancellation())
        return;

    auto level = MessageLevel::Error;
    auto message = blockedTrackerErrorMessage(error);
    if (message.isEmpty())
        message = makeString("Failed to load resource"_s, error.localizedDescription().isEmpty() ? ""_s : ": "_s, error.localizedDescription());
    else
        level = MessageLevel::Info;

    addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::Network, MessageType::Log, level, WTFMove(message), error.failingURL().string(), 0, 0, nullptr, requestIdentifier.toUInt64()));
}

} // namespace WebCore
