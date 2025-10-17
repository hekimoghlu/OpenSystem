/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#import "config.h"
#import "WKHTTPCookieStoreInternal.h"

#import <WebCore/Cookie.h>
#import <WebCore/HTTPCookieAcceptPolicy.h>
#import <WebCore/HTTPCookieAcceptPolicyCocoa.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/HashMap.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/URL.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

static NSArray<NSHTTPCookie *> *coreCookiesToNSCookies(const Vector<WebCore::Cookie>& coreCookies)
{
    return createNSArray(coreCookies, [] (auto& cookie) -> NSHTTPCookie * {
        return cookie;
    }).autorelease();
}

class WKHTTPCookieStoreObserver : public API::HTTPCookieStoreObserver {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WKHTTPCookieStoreObserver);
public:
    static RefPtr<WKHTTPCookieStoreObserver> create(id<WKHTTPCookieStoreObserver> observer)
    {
        return adoptRef(new WKHTTPCookieStoreObserver(observer));
    }

private:
    explicit WKHTTPCookieStoreObserver(id<WKHTTPCookieStoreObserver> observer)
        : m_observer(observer)
    {
    }

    void cookiesDidChange(API::HTTPCookieStore& cookieStore) final
    {
        if ([m_observer respondsToSelector:@selector(cookiesDidChangeInCookieStore:)])
            [m_observer cookiesDidChangeInCookieStore:wrapper(cookieStore)];
    }

    WeakObjCPtr<id<WKHTTPCookieStoreObserver>> m_observer;
};

@implementation WKHTTPCookieStore {
    HashMap<CFTypeRef, RefPtr<WKHTTPCookieStoreObserver>> _observers;
}

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKHTTPCookieStore.class, self))
        return;

    for (RefPtr observer : _observers.values())
        _cookieStore->unregisterObserver(*observer);

    _cookieStore->API::HTTPCookieStore::~HTTPCookieStore();

    [super dealloc];
}

- (void)getAllCookies:(void (^)(NSArray<NSHTTPCookie *> *))completionHandler
{
    _cookieStore->cookies([handler = adoptNS([completionHandler copy])](const Vector<WebCore::Cookie>& cookies) {
        auto rawHandler = (void (^)(NSArray<NSHTTPCookie *> *))handler.get();
        rawHandler(coreCookiesToNSCookies(cookies));
    });
}

- (void)setCookie:(NSHTTPCookie *)cookie completionHandler:(void (^)(void))completionHandler
{
    _cookieStore->setCookies({ cookie }, [handler = adoptNS([completionHandler copy])]() {
        auto rawHandler = (void (^)())handler.get();
        if (rawHandler)
            rawHandler();
    });

}

- (void)deleteCookie:(NSHTTPCookie *)cookie completionHandler:(void (^)(void))completionHandler
{
    _cookieStore->deleteCookie(cookie, [handler = adoptNS([completionHandler copy])]() {
        auto rawHandler = (void (^)())handler.get();
        if (rawHandler)
            rawHandler();
    });
}

- (void)addObserver:(id<WKHTTPCookieStoreObserver>)observer
{
    auto result = _observers.add((__bridge CFTypeRef)observer, nullptr);
    if (!result.isNewEntry)
        return;

    result.iterator->value = WKHTTPCookieStoreObserver::create(observer);
    _cookieStore->registerObserver(*result.iterator->value);
}

- (void)removeObserver:(id<WKHTTPCookieStoreObserver>)observer
{
    auto result = _observers.take((__bridge CFTypeRef)observer);
    if (!result)
        return;

    _cookieStore->unregisterObserver(*result);
}

static WebCore::HTTPCookieAcceptPolicy toHTTPCookieAcceptPolicy(WKCookiePolicy wkCookiePolicy)
{
    switch (wkCookiePolicy) {
    case WKCookiePolicyAllow:
        return WebCore::HTTPCookieAcceptPolicy::OnlyFromMainDocumentDomain;
    case WKCookiePolicyDisallow:
        return WebCore::HTTPCookieAcceptPolicy::Never;
    }
    ASSERT_NOT_REACHED();
}

static WKCookiePolicy toWKCookiePolicy(WebCore::HTTPCookieAcceptPolicy policy)
{
    switch (policy) {
    case WebCore::HTTPCookieAcceptPolicy::OnlyFromMainDocumentDomain:
        return WKCookiePolicyAllow;
    case WebCore::HTTPCookieAcceptPolicy::Never:
        return WKCookiePolicyDisallow;
    case WebCore::HTTPCookieAcceptPolicy::AlwaysAccept:
    case WebCore::HTTPCookieAcceptPolicy::ExclusivelyFromMainDocumentDomain:
        return WKCookiePolicyAllow;
    }
}

- (void)setCookiePolicy:(WKCookiePolicy)policy completionHandler:(void (^)(void))completionHandler
{
    _cookieStore->setHTTPCookieAcceptPolicy(toHTTPCookieAcceptPolicy(policy), [completionHandler = makeBlockPtr(completionHandler)] {
        if (completionHandler)
            completionHandler.get()();
    });
}

- (void)getCookiePolicy:(void (^)(WKCookiePolicy))completionHandler
{
    _cookieStore->getHTTPCookieAcceptPolicy([completionHandler = makeBlockPtr(completionHandler)] (WebCore::HTTPCookieAcceptPolicy policy) {
        completionHandler(toWKCookiePolicy(policy));
    });
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_cookieStore;
}

@end

@implementation WKHTTPCookieStore (WKPrivate)

- (void)_getCookiesForURL:(NSURL *)url completionHandler:(void (^)(NSArray<NSHTTPCookie *> *))completionHandler
{
    _cookieStore->cookiesForURL(url, [handler = makeBlockPtr(completionHandler)] (const Vector<WebCore::Cookie>& cookies) {
        handler.get()(coreCookiesToNSCookies(cookies));
    });
}

- (void)_setCookieAcceptPolicy:(NSHTTPCookieAcceptPolicy)policy completionHandler:(void (^)())completionHandler
{
    _cookieStore->setHTTPCookieAcceptPolicy(WebCore::toHTTPCookieAcceptPolicy(policy), [completionHandler = makeBlockPtr(completionHandler)] {
        completionHandler.get()();
    });
}

- (void)_flushCookiesToDiskWithCompletionHandler:(void(^)(void))completionHandler
{
    _cookieStore->flushCookies([completionHandler = makeBlockPtr(completionHandler)]() {
        completionHandler();
    });
}

@end
