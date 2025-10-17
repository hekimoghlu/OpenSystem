/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#include "AutomationInstrumentation.h"

#if ENABLE(WEBDRIVER_BIDI)

#include <JavaScriptCore/ConsoleMessage.h>
#include <JavaScriptCore/ConsoleTypes.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Observer.h>
#include <wtf/StdLibExtras.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

using namespace Inspector;

namespace {
static WeakPtr<AutomationInstrumentationClient>& automationClient()
{
    static NeverDestroyed<WeakPtr<AutomationInstrumentationClient>> s_client;
    return s_client.get();
}
}

void AutomationInstrumentation::setClient(const AutomationInstrumentationClient &client)
{
    ASSERT(!automationClient());
    automationClient() = client;
}

void AutomationInstrumentation::clearClient()
{
    automationClient().clear();
}

void AutomationInstrumentation::addMessageToConsole(const std::unique_ptr<ConsoleMessage>& message)
{
    if (LIKELY(!automationClient()))
        return;

    WTF::ensureOnMainThread([source = message->source(), type = message->type(), level = message->level(), messageText = message->message(), timestamp = message->timestamp()] {
        if (RefPtr client = automationClient().get())
            client->addMessageToConsole(source, level, messageText, type, timestamp);
    });
}

} // namespace WebCore

#endif
