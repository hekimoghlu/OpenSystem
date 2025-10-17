/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include "MessageReceiver.h"
#include "NetworkProcessSupplement.h"
#include <pal/SessionID.h>
#include <stdint.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/WeakRef.h>

#if USE(SOUP)
#include "SoupCookiePersistentStorageType.h"
#endif

OBJC_CLASS NSHTTPCookieStorage;

namespace WebCore {
struct Cookie;
enum class HTTPCookieAcceptPolicy : uint8_t;
}

namespace WebKit {

class NetworkProcess;

class WebCookieManager : public NetworkProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebCookieManager);
    WTF_MAKE_NONCOPYABLE(WebCookieManager);
public:
    WebCookieManager(NetworkProcess&);
    ~WebCookieManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName();

    void setHTTPCookieAcceptPolicy(PAL::SessionID, WebCore::HTTPCookieAcceptPolicy, CompletionHandler<void()>&&);

#if USE(SOUP)
    void setCookiePersistentStorage(PAL::SessionID, const String& storagePath, SoupCookiePersistentStorageType);
#endif

    void notifyCookiesDidChange(PAL::SessionID);

private:
    Ref<NetworkProcess> protectedProcess();

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void getHostnamesWithCookies(PAL::SessionID, CompletionHandler<void(Vector<String>&&)>&&);

    void deleteCookie(PAL::SessionID, const WebCore::Cookie&, CompletionHandler<void()>&&);
    void deleteCookiesForHostnames(PAL::SessionID, const Vector<String>&, CompletionHandler<void()>&&);
    void deleteAllCookies(PAL::SessionID, CompletionHandler<void()>&&);
    void deleteAllCookiesModifiedSince(PAL::SessionID, WallTime, CompletionHandler<void()>&&);

    void setCookie(PAL::SessionID, const Vector<WebCore::Cookie>&, CompletionHandler<void()>&&);
    void setCookies(PAL::SessionID, const Vector<WebCore::Cookie>&, const URL&, const URL& mainDocumentURL, CompletionHandler<void()>&&);
    void getAllCookies(PAL::SessionID, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&&);
    void getCookies(PAL::SessionID, const URL&, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&&);

    void platformSetHTTPCookieAcceptPolicy(PAL::SessionID, WebCore::HTTPCookieAcceptPolicy, CompletionHandler<void()>&&);
    void getHTTPCookieAcceptPolicy(PAL::SessionID, CompletionHandler<void(WebCore::HTTPCookieAcceptPolicy)>&&);

#if USE(SOUP)
    void replaceCookies(PAL::SessionID, const Vector<WebCore::Cookie>&, CompletionHandler<void()>&&);
#endif

    void startObservingCookieChanges(PAL::SessionID);
    void stopObservingCookieChanges(PAL::SessionID);

    WeakRef<NetworkProcess> m_process;
};

#if PLATFORM(COCOA)
void saveCookies(NSHTTPCookieStorage *, CompletionHandler<void()>&&);
#endif

} // namespace WebKit
