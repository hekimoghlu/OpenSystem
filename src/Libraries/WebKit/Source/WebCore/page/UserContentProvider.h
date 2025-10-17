/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#include <functional>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

#if ENABLE(CONTENT_EXTENSIONS)
#include "ContentExtensionsBackend.h"
#include "ContentRuleListResults.h"
#endif

namespace WebCore {
class UserContentProviderInvalidationClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::UserContentProviderInvalidationClient> : std::true_type { };
}

namespace WebCore {

class DOMWrapperWorld;
class DocumentLoader;
class Page;
class UserContentProvider;
class UserMessageHandlerDescriptor;
class UserScript;
class UserStyleSheet;

class UserContentProviderInvalidationClient : public CanMakeWeakPtr<UserContentProviderInvalidationClient> {
public:
    virtual ~UserContentProviderInvalidationClient()
    {
    }
    
    virtual void didInvalidate(UserContentProvider&) = 0;
};

class UserContentProvider : public RefCounted<UserContentProvider> {
public:
    WEBCORE_EXPORT UserContentProvider();
    WEBCORE_EXPORT virtual ~UserContentProvider();

    virtual void forEachUserScript(Function<void(DOMWrapperWorld&, const UserScript&)>&&) const = 0;
    virtual void forEachUserStyleSheet(Function<void(const UserStyleSheet&)>&&) const = 0;
#if ENABLE(USER_MESSAGE_HANDLERS)
    virtual void forEachUserMessageHandler(Function<void(const UserMessageHandlerDescriptor&)>&&) const = 0;
#endif
#if ENABLE(CONTENT_EXTENSIONS)
    virtual ContentExtensions::ContentExtensionsBackend& userContentExtensionBackend() = 0;
#endif

    void registerForUserMessageHandlerInvalidation(UserContentProviderInvalidationClient&);
    void unregisterForUserMessageHandlerInvalidation(UserContentProviderInvalidationClient&);

    void addPage(Page&);
    void removePage(Page&);

#if ENABLE(CONTENT_EXTENSIONS)
    ContentRuleListResults processContentRuleListsForLoad(Page&, const URL&, OptionSet<ContentExtensions::ResourceType>, DocumentLoader& initiatingDocumentLoader, const URL& redirectFrom = { });
#endif

protected:
    WEBCORE_EXPORT void invalidateAllRegisteredUserMessageHandlerInvalidationClients();
    WEBCORE_EXPORT void invalidateInjectedStyleSheetCacheInAllFramesInAllPages();

private:
    WeakHashSet<Page> m_pages;
    WeakHashSet<UserContentProviderInvalidationClient> m_userMessageHandlerInvalidationClients;
};

} // namespace WebCore
