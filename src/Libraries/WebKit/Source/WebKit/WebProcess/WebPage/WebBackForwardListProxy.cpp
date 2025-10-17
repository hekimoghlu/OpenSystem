/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#include "WebBackForwardListProxy.h"

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "SessionState.h"
#include "SessionStateConversion.h"
#include "WebHistoryItemClient.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include "WebProcessProxyMessages.h"
#include <WebCore/BackForwardCache.h>
#include <WebCore/HistoryController.h>
#include <WebCore/HistoryItem.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/ProcessID.h>

namespace WebKit {
using namespace WebCore;

void WebBackForwardListProxy::removeItem(BackForwardItemIdentifier itemID)
{
    BackForwardCache::singleton().remove(itemID);
    WebCore::Page::clearPreviousItemFromAllPages(itemID);
}

WebBackForwardListProxy::WebBackForwardListProxy(WebPage& page)
    : m_page(&page)
{
}

void WebBackForwardListProxy::addItem(Ref<HistoryItem>&& item)
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    LOG(BackForward, "(Back/Forward) WebProcess pid %i setting item %p for id %s with url %s", getCurrentProcessID(), item.ptr(), item->itemID().toString().utf8().data(), item->urlString().utf8().data());
    m_cachedBackForwardListCounts = std::nullopt;
    page->send(Messages::WebPageProxy::BackForwardAddItem(toFrameState(item.get())));
}

void WebBackForwardListProxy::setChildItem(BackForwardFrameItemIdentifier frameItemID, Ref<HistoryItem>&& item)
{
    if (RefPtr page = m_page.get())
        page->send(Messages::WebPageProxy::BackForwardSetChildItem(frameItemID, toFrameState(item)));
}

void WebBackForwardListProxy::goToItem(HistoryItem& item)
{
    if (!m_page)
        return;

    auto sendResult = m_page->sendSync(Messages::WebPageProxy::BackForwardGoToItem(item.itemID()));
    auto [backForwardListCounts] = sendResult.takeReplyOr(WebBackForwardListCounts { });
    m_cachedBackForwardListCounts = backForwardListCounts;
}

RefPtr<HistoryItem> WebBackForwardListProxy::itemAtIndex(int itemIndex, FrameIdentifier frameID)
{
    RefPtr page = m_page.get();
    if (!page)
        return nullptr;

    auto sendResult = page->sendSync(Messages::WebPageProxy::BackForwardItemAtIndex(itemIndex, frameID));
    auto [frameState] = sendResult.takeReplyOr(nullptr);
    if (!frameState)
        return nullptr;

    Ref historyItemClient = page->historyItemClient();
    auto ignoreHistoryItemChangesForScope = historyItemClient->ignoreChangesForScope();
    return toHistoryItem(historyItemClient, *frameState);
}

unsigned WebBackForwardListProxy::backListCount() const
{
    return cacheListCountsIfNecessary().backCount;
}

unsigned WebBackForwardListProxy::forwardListCount() const
{
    return cacheListCountsIfNecessary().forwardCount;
}

bool WebBackForwardListProxy::containsItem(const WebCore::HistoryItem& item) const
{
    auto sendResult = m_page->sendSync(Messages::WebPageProxy::BackForwardListContainsItem(item.itemID()), m_page->identifier());
    auto [contains] = sendResult.takeReplyOr(false);
    return contains;
}

const WebBackForwardListCounts& WebBackForwardListProxy::cacheListCountsIfNecessary() const
{
    if (!m_cachedBackForwardListCounts) {
        WebBackForwardListCounts backForwardListCounts;
        if (m_page) {
            auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebPageProxy::BackForwardListCounts(), m_page->identifier());
            if (sendResult.succeeded())
                std::tie(backForwardListCounts) = sendResult.takeReply();
        }
        m_cachedBackForwardListCounts = backForwardListCounts;
    }
    return *m_cachedBackForwardListCounts;
}

void WebBackForwardListProxy::clearCachedListCounts()
{
    m_cachedBackForwardListCounts = WebBackForwardListCounts { };
}

void WebBackForwardListProxy::close()
{
    ASSERT(m_page);
    m_page = nullptr;
    m_cachedBackForwardListCounts = WebBackForwardListCounts { };
}

} // namespace WebKit
