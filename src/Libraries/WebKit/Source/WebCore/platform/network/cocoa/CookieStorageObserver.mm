/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#import "CookieStorageObserver.h"

#import <pal/spi/cocoa/NSURLConnectionSPI.h>
#import <wtf/MainThread.h>
#import <wtf/ProcessPrivilege.h>
#import <wtf/TZoneMallocInlines.h>

@interface WebNSHTTPCookieStorageDummyForInternalAccess : NSObject {
@public
    NSHTTPCookieStorageInternal *_internal;
}
@end

@implementation WebNSHTTPCookieStorageDummyForInternalAccess
@end

@interface NSHTTPCookieStorageInternal : NSObject
- (void)registerForPostingNotificationsWithContext:(NSHTTPCookieStorage *)context;
@end

@interface WebCookieObserverAdapter : NSObject {
    WebCore::CookieStorageObserver* observer;
}
- (instancetype)initWithObserver:(WebCore::CookieStorageObserver&)theObserver;
- (void)cookiesChangedNotificationHandler:(NSNotification *)notification;

@end

@implementation WebCookieObserverAdapter

- (instancetype)initWithObserver:(WebCore::CookieStorageObserver&)theObserver
{
    self = [super init];
    if (!self)
        return nil;

    observer = &theObserver;

    return self;
}

- (void)cookiesChangedNotificationHandler:(NSNotification *)notification
{
    UNUSED_PARAM(notification);
    observer->cookiesDidChange();
}

@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CookieStorageObserver);

CookieStorageObserver::CookieStorageObserver(NSHTTPCookieStorage *cookieStorage)
    : m_cookieStorage(cookieStorage)
{
    ASSERT(isMainThread());
    ASSERT(m_cookieStorage);
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));
}

CookieStorageObserver::~CookieStorageObserver()
{
    ASSERT(isMainThread());

    if (m_cookieChangeCallback) {
        ASSERT(m_observerAdapter);
        stopObserving();
    }
}

void CookieStorageObserver::startObserving(WTF::Function<void()>&& callback)
{
    ASSERT(isMainThread());
    ASSERT(!m_cookieChangeCallback);
    ASSERT(!m_observerAdapter);
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));

    m_cookieChangeCallback = WTFMove(callback);
    m_observerAdapter = adoptNS([[WebCookieObserverAdapter alloc] initWithObserver:*this]);

    if (!m_hasRegisteredInternalsForNotifications) {
        if (m_cookieStorage.get() != [NSHTTPCookieStorage sharedHTTPCookieStorage]) {
            auto internalObject = (static_cast<WebNSHTTPCookieStorageDummyForInternalAccess *>(m_cookieStorage.get()))->_internal;
            [internalObject registerForPostingNotificationsWithContext:m_cookieStorage.get()];
        }

        m_hasRegisteredInternalsForNotifications = true;
    }

    [[NSNotificationCenter defaultCenter] addObserver:m_observerAdapter.get() selector:@selector(cookiesChangedNotificationHandler:) name:NSHTTPCookieManagerCookiesChangedNotification object:m_cookieStorage.get()];
}

void CookieStorageObserver::stopObserving()
{
    ASSERT(isMainThread());
    ASSERT(m_cookieChangeCallback);
    ASSERT(m_observerAdapter);
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));

    [[NSNotificationCenter defaultCenter] removeObserver:m_observerAdapter.get() name:NSHTTPCookieManagerCookiesChangedNotification object:nil];

    m_cookieChangeCallback = nullptr;
    m_observerAdapter = nil;
}

void CookieStorageObserver::cookiesDidChange()
{
    callOnMainThread([weakThis = WeakPtr { *this }] {
        if (weakThis && weakThis->m_cookieChangeCallback)
            weakThis->m_cookieChangeCallback();
    });
}

} // namespace WebCore
