/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#include "WebProcessSyncClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/Page.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebProcessSyncClient);

WebProcessSyncClient::WebProcessSyncClient(WebPage& webPage)
    : m_page(webPage)
{
}

Ref<WebPage> WebProcessSyncClient::protectedPage() const
{
    return m_page.get();
}

bool WebProcessSyncClient::siteIsolationEnabled()
{
    RefPtr<WebCore::Page> corePage = protectedPage()->protectedCorePage();
    return corePage ? corePage->settings().siteIsolationEnabled() : false;
}

void WebProcessSyncClient::broadcastProcessSyncDataToOtherProcesses(const WebCore::ProcessSyncData& data)
{
    ASSERT(siteIsolationEnabled());
    protectedPage()->send(Messages::WebPageProxy::BroadcastProcessSyncData(data));
}

void WebProcessSyncClient::broadcastTopDocumentSyncDataToOtherProcesses(WebCore::DocumentSyncData& data)
{
    ASSERT(siteIsolationEnabled());
    protectedPage()->send(Messages::WebPageProxy::BroadcastTopDocumentSyncData(data));
}

} // namespace WebKit
