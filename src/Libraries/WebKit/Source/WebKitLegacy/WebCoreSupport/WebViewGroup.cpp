/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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
#include "WebViewGroup.h"

#include "WebStorageNamespaceProvider.h"
#include "WebView.h"
#include "WebVisitedLinkStore.h"
#include <WebCore/UserContentController.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringHash.h>

using namespace WebCore;

// Any named groups will live for the lifetime of the process, thanks to the reference held by the RefPtr.
static HashMap<String, RefPtr<WebViewGroup>>& webViewGroups()
{
    static NeverDestroyed<HashMap<String, RefPtr<WebViewGroup>>> webViewGroups;

    return webViewGroups;
}

Ref<WebViewGroup> WebViewGroup::getOrCreate(const String& name, const String& localStorageDatabasePath)
{
    if (name.isEmpty())
        return adoptRef(*new WebViewGroup(String(), localStorageDatabasePath));

    auto& webViewGroup = webViewGroups().add(name, nullptr).iterator->value;
    if (!webViewGroup) {
        auto result = adoptRef(*new WebViewGroup(name, localStorageDatabasePath));
        webViewGroup = result.copyRef();
        return result;
    }

    if (!webViewGroup->m_storageNamespaceProvider && webViewGroup->m_localStorageDatabasePath.isEmpty() && !localStorageDatabasePath.isEmpty())
        webViewGroup->m_localStorageDatabasePath = localStorageDatabasePath;

    return *webViewGroup;
}

WebViewGroup* WebViewGroup::get(const String& name)
{
    ASSERT(!name.isEmpty());

    return webViewGroups().get(name);
}

WebViewGroup::WebViewGroup(const String& name, const String& localStorageDatabasePath)
    : m_name(name)
    , m_localStorageDatabasePath(localStorageDatabasePath)
    , m_userContentController(UserContentController::create())
    , m_visitedLinkStore(WebVisitedLinkStore::create())
{
}

WebViewGroup::~WebViewGroup()
{
    ASSERT(m_name.isEmpty());
    ASSERT(m_webViews.isEmpty());
}

void WebViewGroup::addWebView(WebView *webView)
{
    ASSERT(!m_webViews.contains(webView));

    m_webViews.add(webView);
}

void WebViewGroup::removeWebView(WebView *webView)
{
    ASSERT(m_webViews.contains(webView));

    m_webViews.remove(webView);
}

StorageNamespaceProvider& WebViewGroup::storageNamespaceProvider()
{
    if (!m_storageNamespaceProvider)
        m_storageNamespaceProvider = WebKit::WebStorageNamespaceProvider::create(m_localStorageDatabasePath);

    return *m_storageNamespaceProvider;
}
