/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#include <WebCore/Cookie.h>
#include <pal/SessionID.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

#if USE(SOUP)
#include "SoupCookiePersistentStorageType.h"
#endif

namespace WebCore {
struct Cookie;
enum class HTTPCookieAcceptPolicy : uint8_t;
}

namespace WebKit {
class NetworkProcessProxy;
class WebsiteDataStore;
}

namespace API {

class HTTPCookieStore;

class HTTPCookieStoreObserver : public RefCountedAndCanMakeWeakPtr<HTTPCookieStoreObserver> {
public:
    virtual ~HTTPCookieStoreObserver() { }
    virtual void cookiesDidChange(HTTPCookieStore&) = 0;
};

class HTTPCookieStore final : public ObjectImpl<Object::Type::HTTPCookieStore> {
public:
    static Ref<HTTPCookieStore> create(WebKit::WebsiteDataStore& websiteDataStore)
    {
        return adoptRef(*new HTTPCookieStore(websiteDataStore));
    }

    virtual ~HTTPCookieStore();

    void cookies(CompletionHandler<void(Vector<WebCore::Cookie>&&)>&&);
    void cookiesForURL(WTF::URL&&, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&&);
    void setCookies(Vector<WebCore::Cookie>&&, CompletionHandler<void()>&&);
    void deleteCookie(const WebCore::Cookie&, CompletionHandler<void()>&&);
    void deleteCookiesForHostnames(const Vector<WTF::String>&, CompletionHandler<void()>&&);
    
    void deleteAllCookies(CompletionHandler<void()>&&);
    void setHTTPCookieAcceptPolicy(WebCore::HTTPCookieAcceptPolicy, CompletionHandler<void()>&&);
    void getHTTPCookieAcceptPolicy(CompletionHandler<void(const WebCore::HTTPCookieAcceptPolicy&)>&&);
    void flushCookies(CompletionHandler<void()>&&);

    void registerObserver(HTTPCookieStoreObserver&);
    void unregisterObserver(HTTPCookieStoreObserver&);

    void cookiesDidChange();

    void filterAppBoundCookies(Vector<WebCore::Cookie>&&, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&&);

#if USE(SOUP)
    void replaceCookies(Vector<WebCore::Cookie>&&, CompletionHandler<void()>&&);
    void getAllCookies(CompletionHandler<void(const Vector<WebCore::Cookie>&)>&&);

    void setCookiePersistentStorage(const WTF::String& storagePath, WebKit::SoupCookiePersistentStorageType);
#endif

private:
    HTTPCookieStore(WebKit::WebsiteDataStore&);
    WebKit::NetworkProcessProxy* networkProcessIfExists();
    WebKit::NetworkProcessProxy* networkProcessLaunchingIfNecessary();

    PAL::SessionID m_sessionID;
    WeakPtr<WebKit::WebsiteDataStore> m_owningDataStore;
    WeakHashSet<HTTPCookieStoreObserver> m_observers;
};

}
