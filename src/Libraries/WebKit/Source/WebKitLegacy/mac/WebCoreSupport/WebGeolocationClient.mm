/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#import "WebGeolocationClient.h"

#if ENABLE(GEOLOCATION)

#import "WebDelegateImplementationCaching.h"
#import "WebFrameInternal.h"
#import "WebGeolocationPositionInternal.h"
#import "WebSecurityOriginInternal.h"
#import "WebUIDelegatePrivate.h"
#import "WebViewInternal.h"
#import <WebCore/Document.h>
#import <WebCore/Geolocation.h>
#import <WebCore/LocalFrame.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/NakedPtr.h>
#import <wtf/NakedRef.h>
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKResponder.h>
#import <WebKitLegacy/WebCoreThreadRun.h>
#endif

using namespace WebCore;

#if !PLATFORM(IOS_FAMILY)
@interface WebGeolocationPolicyListener : NSObject <WebAllowDenyPolicyListener>
{
    RefPtr<Geolocation> _geolocation;
}
- (id)initWithGeolocation:(NakedRef<Geolocation>)geolocation;
@end
#else
@interface WebGeolocationPolicyListener : NSObject <WebAllowDenyPolicyListener>
{
    RefPtr<Geolocation> _geolocation;
    RetainPtr<WebView> _webView;
}
- (id)initWithGeolocation:(NakedPtr<Geolocation>)geolocation forWebView:(WebView*)webView;
@end
#endif

#if PLATFORM(IOS_FAMILY)
@interface WebGeolocationProviderInitializationListener : NSObject <WebGeolocationProviderInitializationListener> {
@private
    RefPtr<Geolocation> m_geolocation;
}
- (id)initWithGeolocation:(NakedRef<Geolocation>)geolocation;
@end
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebGeolocationClient);

WebGeolocationClient::WebGeolocationClient(WebView *webView)
    : m_webView(webView)
{
}

void WebGeolocationClient::geolocationDestroyed()
{
    delete this;
}

void WebGeolocationClient::startUpdating(const String& authorizationToken, bool enableHighAccuracy)
{
    UNUSED_PARAM(authorizationToken);
#if PLATFORM(IOS_FAMILY)
    if (enableHighAccuracy)
        setEnableHighAccuracy(true);
#else
    UNUSED_PARAM(enableHighAccuracy);
#endif

    [[m_webView _geolocationProvider] registerWebView:m_webView];
}

void WebGeolocationClient::stopUpdating()
{
    [[m_webView _geolocationProvider] unregisterWebView:m_webView];
}

#if PLATFORM(IOS_FAMILY)
void WebGeolocationClient::setEnableHighAccuracy(bool wantsHighAccuracy)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [[m_webView _geolocationProvider] setEnableHighAccuracy:wantsHighAccuracy];
    END_BLOCK_OBJC_EXCEPTIONS
}
#endif

void WebGeolocationClient::requestPermission(Geolocation& geolocation)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    SEL selector = @selector(webView:decidePolicyForGeolocationRequestFromOrigin:frame:listener:);
    if (![[m_webView UIDelegate] respondsToSelector:selector]) {
        geolocation.setIsAllowed(false, { });
        return;
    }

#if !PLATFORM(IOS_FAMILY)
    auto* frame = geolocation.frame();

    if (!frame) {
        geolocation.setIsAllowed(false, { });
        return;
    }

    auto webOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:&frame->document()->securityOrigin()]);
    auto listener = adoptNS([[WebGeolocationPolicyListener alloc] initWithGeolocation:geolocation]);

    CallUIDelegate(m_webView, selector, webOrigin.get(), kit(frame), listener.get());
#else
    RetainPtr<WebGeolocationProviderInitializationListener> listener = adoptNS([[WebGeolocationProviderInitializationListener alloc] initWithGeolocation:geolocation]);
    [[m_webView _geolocationProvider] initializeGeolocationForWebView:m_webView listener:listener.get()];
#endif
    END_BLOCK_OBJC_EXCEPTIONS
}

std::optional<GeolocationPositionData> WebGeolocationClient::lastPosition()
{
    return core([[m_webView _geolocationProvider] lastPosition]);
}

#if !PLATFORM(IOS_FAMILY)
@implementation WebGeolocationPolicyListener

- (id)initWithGeolocation:(NakedRef<Geolocation>)geolocation
{
    if (!(self = [super init]))
        return nil;
    _geolocation = geolocation.ptr();
    return self;
}

- (void)allow
{
    _geolocation->setIsAllowed(true, { });
}

- (void)deny
{
    _geolocation->setIsAllowed(false, { });
}

@end

#else
@implementation WebGeolocationPolicyListener
- (id)initWithGeolocation:(NakedPtr<Geolocation>)geolocation forWebView:(WebView*)webView
{
    self = [super init];
    if (!self)
        return nil;
    _geolocation = geolocation.get();
    _webView = webView;
    return self;
}

- (void)allow
{
    WebThreadRun(^{
        _geolocation->setIsAllowed(true, { });
    });
}

- (void)deny
{
    WebThreadRun(^{
        _geolocation->setIsAllowed(false, { });
    });
}

- (void)denyOnlyThisRequest
{
    WebThreadRun(^{
        // A soft deny does not prevent subsequent request from the Geolocation object.
        [self deny];
        _geolocation->resetIsAllowed();
    });
}

- (BOOL)shouldClearCache
{
    // Theoretically, WebView could changes the WebPreferences after we get the pointer.
    // We lock to be on the safe side.
    WebThreadLock();

    return [[_webView.get() preferences] _alwaysRequestGeolocationPermission];
}
@end

@implementation WebGeolocationProviderInitializationListener
- (id)initWithGeolocation:(NakedRef<Geolocation>)geolocation
{
    self = [super init];
    if (self)
        m_geolocation = geolocation.ptr();
    return self;
}

- (void)initializationAllowedWebView:(WebView *)webView
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    auto* frame = m_geolocation->frame();
    if (!frame)
        return;
    auto webOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:&frame->document()->securityOrigin()]);
    auto listener = adoptNS([[WebGeolocationPolicyListener alloc] initWithGeolocation:m_geolocation.get() forWebView:webView]);
    SEL selector = @selector(webView:decidePolicyForGeolocationRequestFromOrigin:frame:listener:);
    CallUIDelegate(webView, selector, webOrigin.get(), kit(frame), listener.get());

    END_BLOCK_OBJC_EXCEPTIONS
}

- (void)initializationDeniedWebView:(WebView *)webView
{
    m_geolocation->setIsAllowed(false, { });
}
@end
#endif // PLATFORM(IOS_FAMILY)

#endif // ENABLE(GEOLOCATION)
