/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#import "WebVisitedLinkStore.h"

#import "WebDelegateImplementationCaching.h"
#import "WebFrameInternal.h"
#import "WebHistoryDelegate.h"
#import "WebHistoryInternal.h"
#import "WebViewInternal.h"
#import <WebCore/BackForwardCache.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/NeverDestroyed.h>

using namespace WebCore;

static bool s_shouldTrackVisitedLinks;

static HashSet<WeakRef<WebVisitedLinkStore>>& visitedLinkStores()
{
    static NeverDestroyed<HashSet<WeakRef<WebVisitedLinkStore>>> visitedLinkStores;

    return visitedLinkStores;
}


Ref<WebVisitedLinkStore> WebVisitedLinkStore::create()
{
    return adoptRef(*new WebVisitedLinkStore);
}

WebVisitedLinkStore::WebVisitedLinkStore()
    : m_visitedLinksPopulated(false)
{
    visitedLinkStores().add(*this);
}

WebVisitedLinkStore::~WebVisitedLinkStore()
{
    visitedLinkStores().remove(*this);
}

void WebVisitedLinkStore::setShouldTrackVisitedLinks(bool shouldTrackVisitedLinks)
{
    if (s_shouldTrackVisitedLinks == shouldTrackVisitedLinks)
        return;
    s_shouldTrackVisitedLinks = shouldTrackVisitedLinks;
    if (!s_shouldTrackVisitedLinks)
        removeAllVisitedLinks();
}

void WebVisitedLinkStore::removeAllVisitedLinks()
{
    for (auto& visitedLinkStore : visitedLinkStores())
        Ref { visitedLinkStore.get() }->removeVisitedLinkHashes();
}

void WebVisitedLinkStore::addVisitedLink(NSString *urlString)
{
    if (!s_shouldTrackVisitedLinks)
        return;

    size_t length = urlString.length;

    if (auto characters = CFStringGetCharactersPtr((__bridge CFStringRef)urlString)) {
        addVisitedLinkHash(computeSharedStringHash(std::span { reinterpret_cast<const UChar*>(characters), length }));
        return;
    }

    Vector<UniChar, 512> buffer(length);
    [urlString getCharacters:buffer.data()];

    addVisitedLinkHash(computeSharedStringHash(std::span { reinterpret_cast<const UChar*>(buffer.data()), length }));
}

void WebVisitedLinkStore::removeVisitedLink(NSString *urlString)
{
    auto linkHash = computeSharedStringHash(urlString);

    ASSERT(m_visitedLinkHashes.contains(linkHash));
    m_visitedLinkHashes.remove(linkHash);

    invalidateStylesForLink(linkHash);
}

bool WebVisitedLinkStore::isLinkVisited(Page& page, SharedStringHash linkHash, const URL& baseURL, const AtomString& attributeURL)
{
    populateVisitedLinksIfNeeded(page);

    return m_visitedLinkHashes.contains(linkHash);
}

void WebVisitedLinkStore::addVisitedLink(Page& sourcePage, SharedStringHash linkHash)
{
    if (!s_shouldTrackVisitedLinks)
        return;

    addVisitedLinkHash(linkHash);
}

void WebVisitedLinkStore::populateVisitedLinksIfNeeded(Page& page)
{
    if (m_visitedLinksPopulated)
        return;

    m_visitedLinksPopulated = true;

    WebView *webView = kit(&page);
    ASSERT(webView);

    if (webView.historyDelegate) {
        WebHistoryDelegateImplementationCache* implementations = WebViewGetHistoryDelegateImplementations(webView);

        if (implementations->populateVisitedLinksFunc)
            CallHistoryDelegate(implementations->populateVisitedLinksFunc, webView, @selector(populateVisitedLinksForWebView:));

        return;
    }

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [[WebHistory optionalSharedHistory] _addVisitedLinksToVisitedLinkStore:*this];
    END_BLOCK_OBJC_EXCEPTIONS
}

void WebVisitedLinkStore::addVisitedLinkHash(SharedStringHash linkHash)
{
    ASSERT(s_shouldTrackVisitedLinks);

    m_visitedLinkHashes.add(linkHash);

    invalidateStylesForLink(linkHash);
}

void WebVisitedLinkStore::removeVisitedLinkHashes()
{
    m_visitedLinksPopulated = false;
    if (m_visitedLinkHashes.isEmpty())
        return;
    m_visitedLinkHashes.clear();

    invalidateStylesForAllLinks();
}
