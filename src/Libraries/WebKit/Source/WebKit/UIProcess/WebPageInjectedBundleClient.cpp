/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#include "WebPageInjectedBundleClient.h"

#include "APIMessageListener.h"
#include "WKAPICast.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageInjectedBundleClient);

void WebPageInjectedBundleClient::didReceiveMessageFromInjectedBundle(WebPageProxy* page, const String& messageName, API::Object* messageBody)
{
    if (!m_client.didReceiveMessageFromInjectedBundle)
        return;

    m_client.didReceiveMessageFromInjectedBundle(toAPI(page), toAPI(messageName.impl()), toAPI(messageBody), m_client.base.clientInfo);
}

void WebPageInjectedBundleClient::didReceiveSynchronousMessageFromInjectedBundle(WebPageProxy* page, const String& messageName, API::Object* messageBody, CompletionHandler<void(RefPtr<API::Object>)>&& completionHandler)
{
    if (!m_client.didReceiveSynchronousMessageFromInjectedBundle
        && !m_client.didReceiveSynchronousMessageFromInjectedBundleWithListener)
        return completionHandler(nullptr);

    if (m_client.didReceiveSynchronousMessageFromInjectedBundle) {
        WKTypeRef returnDataRef = nullptr;
        m_client.didReceiveSynchronousMessageFromInjectedBundle(toAPI(page), toAPI(messageName.impl()), toAPI(messageBody), &returnDataRef, m_client.base.clientInfo);
        return completionHandler(adoptRef(toImpl(returnDataRef)));
    }

    m_client.didReceiveSynchronousMessageFromInjectedBundleWithListener(toAPI(page), toAPI(messageName.impl()), toAPI(messageBody), toAPI(API::MessageListener::create(WTFMove(completionHandler)).ptr()), m_client.base.clientInfo);
}

void WebPageInjectedBundleClient::didReceiveAsyncMessageFromInjectedBundle(WebPageProxy* page, const String& messageName, API::Object* messageBody, CompletionHandler<void(RefPtr<API::Object>)>&& completionHandler)
{
    if (!m_client.didReceiveAsyncMessageFromInjectedBundle)
        return completionHandler(nullptr);

    m_client.didReceiveAsyncMessageFromInjectedBundle(toAPI(page), toAPI(messageName.impl()), toAPI(messageBody), toAPI(API::MessageListener::create(WTFMove(completionHandler)).ptr()), m_client.base.clientInfo);
}

} // namespace WebKit
