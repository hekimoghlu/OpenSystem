/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "SessionHost.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/Observer.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/StringBuilder.h>

namespace WebDriver {

#if ENABLE(WEBDRIVER_BIDI)
static WeakHashSet<SessionHost::BrowserTerminatedObserver>& browserTerminatedObservers()
{
    static NeverDestroyed<WeakHashSet<SessionHost::BrowserTerminatedObserver>> observers;
    return observers;
}
#endif

void SessionHost::inspectorDisconnected()
{
    Ref<SessionHost> protectedThis(*this);
    // Browser closed or crashed, finish all pending commands with error.
    for (auto messageID : copyToVector(m_commandRequests.keys())) {
        auto responseHandler = m_commandRequests.take(messageID);
        responseHandler({ nullptr, true });
    }

#if ENABLE(WEBDRIVER_BIDI)
    for (auto& observer : browserTerminatedObservers())
        observer(m_sessionID);
#endif
}

long SessionHost::sendCommandToBackend(const String& command, RefPtr<JSON::Object>&& parameters, Function<void (CommandResponse&&)>&& responseHandler)
{
    if (!isConnected()) {
        responseHandler({ nullptr, true });
        return 0;
    }

    static long lastSequenceID = 0;
    long sequenceID = ++lastSequenceID;
    m_commandRequests.add(sequenceID, WTFMove(responseHandler));
    StringBuilder messageBuilder;
    messageBuilder.append("{\"id\":"_s, sequenceID, ",\"method\":\"Automation."_s, command, '"');
    if (parameters)
        messageBuilder.append(",\"params\":"_s, parameters->toJSONString());
    messageBuilder.append('}');
    sendMessageToBackend(messageBuilder.toString());

    return sequenceID;
}

void SessionHost::dispatchMessage(const String& message)
{
    auto messageValue = JSON::Value::parseJSON(message);
    if (!messageValue)
        return;

    auto messageObject = messageValue->asObject();
    if (!messageObject)
        return;

    auto sequenceID = messageObject->getInteger("id"_s);
    if (!sequenceID) {
#if ENABLE(WEBDRIVER_BIDI)
        dispatchEvent(WTFMove(messageObject));
#endif
        return;
    }

    auto responseHandler = m_commandRequests.take(*sequenceID);
    ASSERT(responseHandler);

    CommandResponse response;
    if (auto errorObject = messageObject->getObject("error"_s)) {
        response.responseObject = WTFMove(errorObject);
        response.isError = true;
    } else if (auto resultObject = messageObject->getObject("result"_s)) {
        if (resultObject->size())
            response.responseObject = WTFMove(resultObject);
    }

    responseHandler(WTFMove(response));
}

bool SessionHost::isRemoteBrowser() const
{
    return m_isRemoteBrowser;
}

#if ENABLE(WEBDRIVER_BIDI)
void SessionHost::addBrowserTerminatedObserver(const BrowserTerminatedObserver& observer)
{
    ASSERT(!browserTerminatedObservers().contains(observer));
    browserTerminatedObservers().add(observer);
}

void SessionHost::removeBrowserTerminatedObserver(const BrowserTerminatedObserver& observer)
{
    browserTerminatedObservers().remove(observer);
}

void SessionHost::dispatchEvent(RefPtr<JSON::Object>&& event)
{
    if (m_eventHandler)
        m_eventHandler->dispatchEvent(WTFMove(event));
}
#endif

} // namespace WebDriver
