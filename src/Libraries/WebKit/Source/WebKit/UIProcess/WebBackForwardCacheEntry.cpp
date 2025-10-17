/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "WebBackForwardCacheEntry.h"

#include "Logging.h"
#include "SuspendedPageProxy.h"
#include "WebBackForwardCache.h"
#include "WebProcessMessages.h"
#include "WebProcessProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

static const Seconds expirationDelay { 30_min };

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebBackForwardCacheEntry);

Ref<WebBackForwardCacheEntry> WebBackForwardCacheEntry::create(WebBackForwardCache& backForwardCache, WebCore::BackForwardItemIdentifier backForwardItemID, WebCore::ProcessIdentifier processIdentifier, RefPtr<SuspendedPageProxy>&& suspendedPage)
{
    return adoptRef(*new WebBackForwardCacheEntry(backForwardCache, backForwardItemID, processIdentifier, WTFMove(suspendedPage)));
}

WebBackForwardCacheEntry::WebBackForwardCacheEntry(WebBackForwardCache& backForwardCache, WebCore::BackForwardItemIdentifier backForwardItemID, WebCore::ProcessIdentifier processIdentifier, RefPtr<SuspendedPageProxy>&& suspendedPage)
    : m_backForwardCache(backForwardCache)
    , m_processIdentifier(processIdentifier)
    , m_backForwardItemID(backForwardItemID)
    , m_suspendedPage(WTFMove(suspendedPage))
    , m_expirationTimer(RunLoop::main(), this, &WebBackForwardCacheEntry::expirationTimerFired)
{
    m_expirationTimer.startOneShot(expirationDelay);
}

WebBackForwardCacheEntry::~WebBackForwardCacheEntry()
{
    if (m_backForwardItemID && !m_suspendedPage) {
        if (auto process = this->process())
            process->sendWithAsyncReply(Messages::WebProcess::ClearCachedPage(*m_backForwardItemID), [] { });
    }
}

WebBackForwardCache* WebBackForwardCacheEntry::backForwardCache() const
{
    return m_backForwardCache.get();
}

Ref<SuspendedPageProxy> WebBackForwardCacheEntry::takeSuspendedPage()
{
    ASSERT(m_suspendedPage);
    m_backForwardItemID = std::nullopt;
    m_expirationTimer.stop();
    return std::exchange(m_suspendedPage, nullptr).releaseNonNull();
}

RefPtr<WebProcessProxy> WebBackForwardCacheEntry::process() const
{
    auto process = WebProcessProxy::processForIdentifier(m_processIdentifier);
    ASSERT(process);
    ASSERT(!m_suspendedPage || process == &m_suspendedPage->process());
    return process;
}

void WebBackForwardCacheEntry::expirationTimerFired()
{
    ASSERT(m_backForwardItemID);
    RELEASE_LOG(BackForwardCache, "%p - WebBackForwardCacheEntry::expirationTimerFired backForwardItemID=%s, hasSuspendedPage=%d", this, m_backForwardItemID->toString().utf8().data(), !!m_suspendedPage);
    auto* item = WebBackForwardListItem::itemForID(*m_backForwardItemID);
    ASSERT(item);
    if (RefPtr backForwardCache = m_backForwardCache.get())
        backForwardCache->removeEntry(*item);
}

} // namespace WebKit
