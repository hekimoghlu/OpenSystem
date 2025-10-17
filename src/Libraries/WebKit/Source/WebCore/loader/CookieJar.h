/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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

#include "FrameIdentifier.h"
#include "PageIdentifier.h"
#include "SameSiteInfo.h"
#include <optional>
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

enum class IncludeSecureCookies : bool { No, Yes };
enum class IncludeHttpOnlyCookies : bool { No, Yes };
enum class SecureCookiesAccessed : bool { No, Yes };

class Document;
struct Cookie;
class CookieChangeListener;
struct CookieRequestHeaderFieldProxy;
struct CookieStoreGetOptions;
class NetworkStorageSession;
class StorageSessionProvider;
struct SameSiteInfo;
enum class ShouldPartitionCookie : bool;

class WEBCORE_EXPORT CookieJar : public RefCountedAndCanMakeWeakPtr<CookieJar> {
public:
    static Ref<CookieJar> create(Ref<StorageSessionProvider>&&);
    
    static CookieRequestHeaderFieldProxy cookieRequestHeaderFieldProxy(const Document&, const URL&);

    String cookieRequestHeaderFieldValue(Document&, const URL&) const;

    // These two functions implement document.cookie API, with special rules for HttpOnly cookies.
    virtual String cookies(Document&, const URL&) const;
    virtual void setCookies(Document&, const URL&, const String& cookieString);

    virtual bool cookiesEnabled(Document&);
    virtual void remoteCookiesEnabled(const Document&, CompletionHandler<void(bool)>&&) const;
    virtual std::pair<String, SecureCookiesAccessed> cookieRequestHeaderFieldValue(const URL& firstParty, const SameSiteInfo&, const URL&, std::optional<FrameIdentifier>, std::optional<PageIdentifier>, IncludeSecureCookies) const;
    virtual bool getRawCookies(Document&, const URL&, Vector<Cookie>&) const;
    virtual void setRawCookie(const Document&, const Cookie&, ShouldPartitionCookie);
    virtual void deleteCookie(const Document&, const URL&, const String& cookieName, CompletionHandler<void()>&&);

    virtual void getCookiesAsync(Document&, const URL&, const CookieStoreGetOptions&, CompletionHandler<void(std::optional<Vector<Cookie>>&&)>&&) const;
    virtual void setCookieAsync(Document&, const URL&, const Cookie&, CompletionHandler<void(bool)>&&) const;

#if HAVE(COOKIE_CHANGE_LISTENER_API)
    virtual void addChangeListener(const WebCore::Document&, const CookieChangeListener&);
    virtual void removeChangeListener(const String& host, const CookieChangeListener&);
#endif

    // Cookie Cache.
    virtual void clearCache() { }
    virtual void clearCacheForHost(const String&) { }

    virtual ~CookieJar();
protected:
    static SameSiteInfo sameSiteInfo(const Document&, IsForDOMCookieAccess = IsForDOMCookieAccess::No);
    static IncludeSecureCookies shouldIncludeSecureCookies(const Document&, const URL&);
    CookieJar(Ref<StorageSessionProvider>&&);

private:
    Ref<StorageSessionProvider> m_storageSessionProvider;
    Ref<StorageSessionProvider> protectedStorageSessionProvider() const;
};

} // namespace WebCore
