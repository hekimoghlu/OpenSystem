/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "WebBackForwardListItem.h"

#include "SuspendedPageProxy.h"
#include "WebBackForwardCache.h"
#include "WebBackForwardCacheEntry.h"
#include "WebBackForwardListFrameItem.h"
#include "WebFrameProxy.h"
#include "WebProcessPool.h"
#include "WebProcessProxy.h"
#include <wtf/DebugUtilities.h>
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>

namespace WebKit {
using namespace WebCore;

Ref<WebBackForwardListItem> WebBackForwardListItem::create(Ref<FrameState>&& mainFrameState, WebPageProxyIdentifier pageID, std::optional<FrameIdentifier> navigatedFrameID)
{
    RELEASE_ASSERT(RunLoop::isMain());
    return adoptRef(*new WebBackForwardListItem(WTFMove(mainFrameState), pageID, navigatedFrameID));
}

WebBackForwardListItem::WebBackForwardListItem(Ref<FrameState>&& mainFrameState, WebPageProxyIdentifier pageID, std::optional<FrameIdentifier> navigatedFrameID)
    : m_identifier(*mainFrameState->itemID)
    , m_mainFrameItem(WebBackForwardListFrameItem::create(*this, nullptr, WTFMove(mainFrameState)))
    , m_navigatedFrameID(navigatedFrameID)
    , m_pageID(pageID)
    , m_lastProcessIdentifier(navigatedFrameItem().identifier().processIdentifier())
{
    auto result = allItems().add(m_identifier, *this);
    ASSERT_UNUSED(result, result.isNewEntry);
}

WebBackForwardListItem::~WebBackForwardListItem()
{
    RELEASE_ASSERT(RunLoop::isMain());
    ASSERT(allItems().get(m_identifier) == this);
    allItems().remove(m_identifier);
    removeFromBackForwardCache();
}

HashMap<BackForwardItemIdentifier, WeakRef<WebBackForwardListItem>>& WebBackForwardListItem::allItems()
{
    RELEASE_ASSERT(RunLoop::isMain());
    static NeverDestroyed<HashMap<BackForwardItemIdentifier, WeakRef<WebBackForwardListItem>>> items;
    return items;
}

WebBackForwardListItem* WebBackForwardListItem::itemForID(BackForwardItemIdentifier identifier)
{
    return allItems().get(identifier);
}

static const FrameState* childItemWithDocumentSequenceNumber(const FrameState& frameState, int64_t number)
{
    for (auto& child : frameState.children) {
        if (child->documentSequenceNumber == number)
            return child.ptr();
    }

    return nullptr;
}

static const FrameState* childItemWithTarget(const FrameState& frameState, const String& target)
{
    for (auto& child : frameState.children) {
        if (child->target == target)
            return child.ptr();
    }

    return nullptr;
}

static bool documentTreesAreEqual(const FrameState& a, const FrameState& b)
{
    if (a.documentSequenceNumber != b.documentSequenceNumber)
        return false;

    if (a.children.size() != b.children.size())
        return false;

    for (auto& child : a.children) {
        const FrameState* otherChild = childItemWithDocumentSequenceNumber(b, child->documentSequenceNumber);
        if (!otherChild || !documentTreesAreEqual(child, *otherChild))
            return false;
    }

    return true;
}

bool WebBackForwardListItem::itemIsInSameDocument(const WebBackForwardListItem& other) const
{
    if (m_pageID != other.m_pageID)
        return false;

    // The following logic must be kept in sync with WebCore::HistoryItem::shouldDoSameDocumentNavigationTo().

    Ref mainFrameState = this->mainFrameState();
    Ref otherMainFrameState = other.mainFrameState();

    if (mainFrameState->stateObjectData || otherMainFrameState->stateObjectData)
        return mainFrameState->documentSequenceNumber == otherMainFrameState->documentSequenceNumber;

    URL url = URL({ }, mainFrameState->urlString);
    URL otherURL = URL({ }, otherMainFrameState->urlString);

    if ((url.hasFragmentIdentifier() || otherURL.hasFragmentIdentifier()) && equalIgnoringFragmentIdentifier(url, otherURL))
        return mainFrameState->documentSequenceNumber == otherMainFrameState->documentSequenceNumber;

    return documentTreesAreEqual(mainFrameState, otherMainFrameState);
}

static bool hasSameFrames(const FrameState& a, const FrameState& b)
{
    if (a.target != b.target)
        return false;

    if (a.children.size() != b.children.size())
        return false;

    for (auto& child : a.children) {
        if (!childItemWithTarget(b, child->target))
            return false;
    }

    return true;
}

bool WebBackForwardListItem::itemIsClone(const WebBackForwardListItem& other)
{
    // The following logic must be kept in sync with WebCore::HistoryItem::itemsAreClones().

    if (this == &other)
        return false;

    Ref mainFrameState = this->mainFrameState();
    Ref otherMainFrameState = other.mainFrameState();

    if (mainFrameState->itemSequenceNumber != otherMainFrameState->itemSequenceNumber)
        return false;

    return hasSameFrames(mainFrameState, otherMainFrameState);
}

void WebBackForwardListItem::wasRemovedFromBackForwardList()
{
    removeFromBackForwardCache();
}

void WebBackForwardListItem::removeFromBackForwardCache()
{
    if (RefPtr backForwardCacheEntry = m_backForwardCacheEntry) {
        if (RefPtr backForwardCache = backForwardCacheEntry->backForwardCache())
            backForwardCache->removeEntry(*this);
    }
    ASSERT(!m_backForwardCacheEntry);
}

RefPtr<WebBackForwardCacheEntry> WebBackForwardListItem::protectedBackForwardCacheEntry() const
{
    return m_backForwardCacheEntry;
}

void WebBackForwardListItem::setBackForwardCacheEntry(RefPtr<WebBackForwardCacheEntry>&& backForwardCacheEntry)
{
    m_backForwardCacheEntry = WTFMove(backForwardCacheEntry);
}

SuspendedPageProxy* WebBackForwardListItem::suspendedPage() const
{
    return m_backForwardCacheEntry ? m_backForwardCacheEntry->suspendedPage() : nullptr;
}

Ref<FrameState> WebBackForwardListItem::navigatedFrameState() const
{
    return protectedNavigatedFrameItem()->copyFrameStateWithChildren();
}

Ref<FrameState> WebBackForwardListItem::mainFrameState() const
{
    return m_mainFrameItem->copyFrameStateWithChildren();
}

const String& WebBackForwardListItem::originalURL() const
{
    if (m_isRemoteFrameNavigation)
        return emptyString();
    return mainFrameItem().frameState().originalURLString;
}

const String& WebBackForwardListItem::url() const
{
    if (m_isRemoteFrameNavigation)
        return emptyString();
    return mainFrameItem().frameState().urlString;
}

const String& WebBackForwardListItem::title() const
{
    if (m_isRemoteFrameNavigation)
        return emptyString();
    return mainFrameItem().frameState().title;
}

bool WebBackForwardListItem::wasCreatedByJSWithoutUserInteraction() const
{
    return navigatedFrameItem().frameState().wasCreatedByJSWithoutUserInteraction;
}

void WebBackForwardListItem::setWasRestoredFromSession()
{
    m_mainFrameItem->setWasRestoredFromSession();
}

WebBackForwardListFrameItem& WebBackForwardListItem::navigatedFrameItem() const
{
    if (RefPtr childItem = m_navigatedFrameID ? m_mainFrameItem->childItemForFrameID(*m_navigatedFrameID) : nullptr)
        return childItem.releaseNonNull();
    return m_mainFrameItem;
}

Ref<WebBackForwardListFrameItem> WebBackForwardListItem::protectedNavigatedFrameItem() const
{
    return navigatedFrameItem();
}

WebBackForwardListFrameItem& WebBackForwardListItem::mainFrameItem() const
{
    return m_mainFrameItem;
}

Ref<WebBackForwardListFrameItem> WebBackForwardListItem::protectedMainFrameItem() const
{
    return m_mainFrameItem;
}

#if !LOG_DISABLED
String WebBackForwardListItem::loggingString()
{
    return makeString("Back/forward item ID "_s, identifier().toString(), ", original URL "_s, originalURL(), ", current URL "_s, url(), m_backForwardCacheEntry ? "(has a back/forward cache entry)"_s : ""_s);
}
#endif // !LOG_DISABLED

} // namespace WebKit
