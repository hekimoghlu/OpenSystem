/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#include "WebHistoryItemClient.h"

#include "MessageSenderInlines.h"
#include "SessionState.h"
#include "SessionStateConversion.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/HistoryItem.h>

namespace WebKit {

WebHistoryItemClient::WebHistoryItemClient(WebPage& page)
    : m_page(page)
{
}

ScopeExit<CompletionHandler<void()>> WebHistoryItemClient::ignoreChangesForScope()
{
    m_shouldIgnoreChanges = true;
    return makeScopeExit(CompletionHandler<void()> { [this, protectedThis = Ref { *this }] {
        m_shouldIgnoreChanges = false;
    } });
}

void WebHistoryItemClient::historyItemChanged(const WebCore::HistoryItem& item)
{
    if (m_shouldIgnoreChanges)
        return;
    if (RefPtr page = m_page.get())
        page->send(Messages::WebPageProxy::BackForwardUpdateItem(toFrameState(item)));
}

void WebHistoryItemClient::clearChildren(const WebCore::HistoryItem& item) const
{
    if (m_shouldIgnoreChanges)
        return;
    if (RefPtr page = m_page.get())
        page->send(Messages::WebPageProxy::BackForwardClearChildren(item.itemID(), item.frameItemID()));
}

}
