/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#pragma once

#include "APIObject.h"
#include "SessionState.h"
#include "WebPageProxyIdentifier.h"
#include "WebsiteDataStore.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Ref.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class SuspendedPageProxy;
class WebBackForwardCache;
class WebBackForwardCacheEntry;
class WebBackForwardListFrameItem;

class WebBackForwardListItem : public API::ObjectImpl<API::Object::Type::BackForwardListItem>, public CanMakeWeakPtr<WebBackForwardListItem> {
public:
    static Ref<WebBackForwardListItem> create(Ref<FrameState>&&, WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>);
    virtual ~WebBackForwardListItem();

    static WebBackForwardListItem* itemForID(WebCore::BackForwardItemIdentifier);
    static HashMap<WebCore::BackForwardItemIdentifier, WeakRef<WebBackForwardListItem>>& allItems();

    WebCore::BackForwardItemIdentifier identifier() const { return m_identifier; }
    WebPageProxyIdentifier pageID() const { return m_pageID; }

    WebCore::ProcessIdentifier lastProcessIdentifier() const { return m_lastProcessIdentifier; }
    void setLastProcessIdentifier(const WebCore::ProcessIdentifier& identifier) { m_lastProcessIdentifier = identifier; }

    Ref<FrameState> navigatedFrameState() const;
    Ref<FrameState> mainFrameState() const;

    const String& originalURL() const;
    const String& url() const;
    const String& title() const;
    bool wasCreatedByJSWithoutUserInteraction() const;

    const URL& resourceDirectoryURL() const { return m_resourceDirectoryURL; }
    void setResourceDirectoryURL(URL&& url) { m_resourceDirectoryURL = WTFMove(url); }
    RefPtr<WebsiteDataStore> dataStoreForWebArchive() const { return m_dataStoreForWebArchive; }
    void setDataStoreForWebArchive(WebsiteDataStore* dataStore) { m_dataStoreForWebArchive = dataStore; }

    bool itemIsInSameDocument(const WebBackForwardListItem&) const;
    bool itemIsClone(const WebBackForwardListItem&);

#if PLATFORM(COCOA) || PLATFORM(GTK)
    ViewSnapshot* snapshot() const { return m_snapshot.get(); }
    void setSnapshot(RefPtr<ViewSnapshot>&& snapshot) { m_snapshot = WTFMove(snapshot); }
#endif

    void wasRemovedFromBackForwardList();

    WebBackForwardCacheEntry* backForwardCacheEntry() const { return m_backForwardCacheEntry.get(); }
    RefPtr<WebBackForwardCacheEntry> protectedBackForwardCacheEntry() const;

    SuspendedPageProxy* suspendedPage() const;

    std::optional<WebCore::FrameIdentifier> navigatedFrameID() const { return m_navigatedFrameID; }

    WebBackForwardListFrameItem& navigatedFrameItem() const;
    Ref<WebBackForwardListFrameItem> protectedNavigatedFrameItem() const;

    WebBackForwardListFrameItem& mainFrameItem() const;
    Ref<WebBackForwardListFrameItem> protectedMainFrameItem() const;

    void setIsRemoteFrameNavigation(bool isRemoteFrameNavigation) { m_isRemoteFrameNavigation = isRemoteFrameNavigation; }
    bool isRemoteFrameNavigation() const { return m_isRemoteFrameNavigation; }

    void setWasRestoredFromSession();

#if !LOG_DISABLED
    String loggingString();
#endif

private:
    WebBackForwardListItem(Ref<FrameState>&&, WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>);

    void removeFromBackForwardCache();

    friend class WebBackForwardCache;
    void setBackForwardCacheEntry(RefPtr<WebBackForwardCacheEntry>&&);

    RefPtr<WebsiteDataStore> m_dataStoreForWebArchive;

    const WebCore::BackForwardItemIdentifier m_identifier;
    const Ref<WebBackForwardListFrameItem> m_mainFrameItem;
    const Markable<WebCore::FrameIdentifier> m_navigatedFrameID;
    URL m_resourceDirectoryURL;
    const WebPageProxyIdentifier m_pageID;
    WebCore::ProcessIdentifier m_lastProcessIdentifier;
    RefPtr<WebBackForwardCacheEntry> m_backForwardCacheEntry;
#if PLATFORM(COCOA) || PLATFORM(GTK)
    RefPtr<ViewSnapshot> m_snapshot;
#endif
    bool m_isRemoteFrameNavigation { false };
};

typedef Vector<Ref<WebBackForwardListItem>> BackForwardListItemVector;

} // namespace WebKit
