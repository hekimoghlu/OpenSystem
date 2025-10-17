/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#include "InjectedBundleClient.h"

#include "InjectedBundle.h"
#include "WKBundleAPICast.h"
#include "WebPage.h"
#include "WebPageGroupProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InjectedBundleClient);

InjectedBundleClient::InjectedBundleClient(const WKBundleClientBase* client)
{
    initialize(client);
}

void InjectedBundleClient::didCreatePage(InjectedBundle& bundle, WebPage& page)
{
    if (!m_client.didCreatePage)
        return;

    m_client.didCreatePage(toAPI(&bundle), toAPI(&page), m_client.base.clientInfo);
}

void InjectedBundleClient::willDestroyPage(InjectedBundle& bundle, WebPage& page)
{
    if (!m_client.willDestroyPage)
        return;

    m_client.willDestroyPage(toAPI(&bundle), toAPI(&page), m_client.base.clientInfo);
}

void InjectedBundleClient::didReceiveMessage(InjectedBundle& bundle, const String& messageName, RefPtr<API::Object>&& messageBody)
{
    if (!m_client.didReceiveMessage)
        return;

    m_client.didReceiveMessage(toAPI(&bundle), toAPI(messageName.impl()), toAPI(messageBody.get()), m_client.base.clientInfo);
}

void InjectedBundleClient::didReceiveMessageToPage(InjectedBundle& bundle, WebPage& page, const String& messageName, RefPtr<API::Object>&& messageBody)
{
    if (!m_client.didReceiveMessageToPage)
        return;

    m_client.didReceiveMessageToPage(toAPI(&bundle), toAPI(&page), toAPI(messageName.impl()), toAPI(messageBody.get()), m_client.base.clientInfo);
}

} // namespace WebKit
