/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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

#include <wtf/HashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class StorageNamespaceProvider;
class UserContentController;
}

class WebVisitedLinkStore;

OBJC_CLASS WebView;

class WebViewGroup : public RefCounted<WebViewGroup> {
public:
    static Ref<WebViewGroup> getOrCreate(const String& name, const String& localStorageDatabasePath);
    ~WebViewGroup();

    static WebViewGroup* get(const String& name);

    void addWebView(WebView *);
    void removeWebView(WebView *);

    WebCore::StorageNamespaceProvider& storageNamespaceProvider();
    WebCore::UserContentController& userContentController() { return m_userContentController.get(); }
    WebVisitedLinkStore& visitedLinkStore() { return m_visitedLinkStore.get(); }

private:
    WebViewGroup(const String& name, const String& localStorageDatabasePath);

    String m_name;
    HashSet<WebView *> m_webViews;

    String m_localStorageDatabasePath;
    RefPtr<WebCore::StorageNamespaceProvider> m_storageNamespaceProvider;

    Ref<WebCore::UserContentController> m_userContentController;
    Ref<WebVisitedLinkStore> m_visitedLinkStore;
};
