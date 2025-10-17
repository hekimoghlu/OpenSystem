/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#import "WKProcessGroupPrivate.h"

#import "APINavigationData.h"
#import "APIProcessPoolConfiguration.h"
#import "WKAPICast.h"
#import "WKBrowsingContextControllerInternal.h"
#import "WKBrowsingContextHistoryDelegate.h"
#import "WKNSString.h"
#import "WKNSURL.h"
#import "WKNavigationDataInternal.h"
#import "WKRetainPtr.h"
#import "WKStringCF.h"
#import "WebFrameProxy.h"
#import "WebProcessPool.h"
#import <wtf/RetainPtr.h>
#import <wtf/StdLibExtras.h>
#import <wtf/WeakObjCPtr.h>

#if PLATFORM(IOS_FAMILY)
#import "WKAPICast.h"
#import "WKGeolocationProviderIOS.h"
#import <WebCore/WebCoreThreadSystemInterface.h>
#endif

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
@implementation WKProcessGroup {
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
    RefPtr<WebKit::WebProcessPool> _processPool;

#if PLATFORM(IOS_FAMILY)
    RetainPtr<WKGeolocationProviderIOS> _geolocationProvider;
#endif // PLATFORM(IOS_FAMILY)
}

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
static void setUpInjectedBundleClient(WKProcessGroup *processGroup, WKContextRef contextRef)
ALLOW_DEPRECATED_DECLARATIONS_END
{
    WKContextInjectedBundleClientV1 injectedBundleClient;
    zeroBytes(injectedBundleClient);

    injectedBundleClient.base.version = 1;
    injectedBundleClient.base.clientInfo = (__bridge void*)processGroup;

    WKContextSetInjectedBundleClient(contextRef, &injectedBundleClient.base);
}

static void didNavigateWithNavigationData(WKContextRef, WKPageRef pageRef, WKNavigationDataRef navigationDataRef, WKFrameRef frameRef, const void*)
{
    if (!WebKit::toImpl(frameRef)->isMainFrame())
        return;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WKBrowsingContextController *controller = [WKBrowsingContextController _browsingContextControllerForPageRef:pageRef];
ALLOW_DEPRECATED_DECLARATIONS_END
    auto historyDelegate = controller->_historyDelegate.get();

    if ([historyDelegate respondsToSelector:@selector(browsingContextController:didNavigateWithNavigationData:)])
        [historyDelegate browsingContextController:controller didNavigateWithNavigationData:wrapper(*WebKit::toImpl(navigationDataRef))];
}

static void didPerformClientRedirect(WKContextRef, WKPageRef pageRef, WKURLRef sourceURLRef, WKURLRef destinationURLRef, WKFrameRef frameRef, const void*)
{
    if (!WebKit::toImpl(frameRef)->isMainFrame())
        return;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WKBrowsingContextController *controller = [WKBrowsingContextController _browsingContextControllerForPageRef:pageRef];
ALLOW_DEPRECATED_DECLARATIONS_END
    auto historyDelegate = controller->_historyDelegate.get();

    if ([historyDelegate respondsToSelector:@selector(browsingContextController:didPerformClientRedirectFromURL:toURL:)])
        [historyDelegate browsingContextController:controller didPerformClientRedirectFromURL:wrapper(*WebKit::toImpl(sourceURLRef)) toURL:wrapper(*WebKit::toImpl(destinationURLRef))];
}

static void didPerformServerRedirect(WKContextRef, WKPageRef pageRef, WKURLRef sourceURLRef, WKURLRef destinationURLRef, WKFrameRef frameRef, const void*)
{
    if (!WebKit::toImpl(frameRef)->isMainFrame())
        return;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WKBrowsingContextController *controller = [WKBrowsingContextController _browsingContextControllerForPageRef:pageRef];
ALLOW_DEPRECATED_DECLARATIONS_END
    auto historyDelegate = controller->_historyDelegate.get();

    if ([historyDelegate respondsToSelector:@selector(browsingContextController:didPerformServerRedirectFromURL:toURL:)])
        [historyDelegate browsingContextController:controller didPerformServerRedirectFromURL:wrapper(*WebKit::toImpl(sourceURLRef)) toURL:wrapper(*WebKit::toImpl(destinationURLRef))];
}

static void didUpdateHistoryTitle(WKContextRef, WKPageRef pageRef, WKStringRef titleRef, WKURLRef urlRef, WKFrameRef frameRef, const void*)
{
    if (!WebKit::toImpl(frameRef)->isMainFrame())
        return;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WKBrowsingContextController *controller = [WKBrowsingContextController _browsingContextControllerForPageRef:pageRef];
ALLOW_DEPRECATED_DECLARATIONS_END
    auto historyDelegate = controller->_historyDelegate.get();

    if ([historyDelegate respondsToSelector:@selector(browsingContextController:didUpdateHistoryTitle:forURL:)])
        [historyDelegate browsingContextController:controller didUpdateHistoryTitle:wrapper(*WebKit::toImpl(titleRef)) forURL:wrapper(*WebKit::toImpl(urlRef))];
}

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
static void setUpHistoryClient(WKProcessGroup *processGroup, WKContextRef contextRef)
ALLOW_DEPRECATED_DECLARATIONS_END
{
    WKContextHistoryClientV0 historyClient;
    zeroBytes(historyClient);

    historyClient.base.version = 0;
    historyClient.base.clientInfo = (__bridge CFTypeRef)processGroup;
    historyClient.didNavigateWithNavigationData = didNavigateWithNavigationData;
    historyClient.didPerformClientRedirect = didPerformClientRedirect;
    historyClient.didPerformServerRedirect = didPerformServerRedirect;
    historyClient.didUpdateHistoryTitle = didUpdateHistoryTitle;

    WKContextSetHistoryClient(contextRef, &historyClient.base);
}

- (id)init
{
    return [self initWithInjectedBundleURL:nil];
}

- (id)initWithInjectedBundleURL:(NSURL *)bundleURL
{
    self = [super init];
    if (!self)
        return nil;

    auto configuration = API::ProcessPoolConfiguration::create();
    configuration->setInjectedBundlePath(bundleURL ? String(bundleURL.path) : String());

    _processPool = WebKit::WebProcessPool::create(configuration);

    setUpInjectedBundleClient(self, toAPI(_processPool.get()));
    setUpHistoryClient(self, toAPI(_processPool.get()));

    return self;
}

- (id)initWithInjectedBundleURL:(NSURL *)bundleURL andCustomClassesForParameterCoder:(NSSet *)classesForCoder
{
    self = [super init];
    if (!self)
        return nil;

    auto configuration = API::ProcessPoolConfiguration::create();
    configuration->setInjectedBundlePath(bundleURL ? String(bundleURL.path) : String());

    _processPool = WebKit::WebProcessPool::create(configuration);

    setUpInjectedBundleClient(self, toAPI(_processPool.get()));
    setUpHistoryClient(self, toAPI(_processPool.get()));

    return self;
}

@end

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
@implementation WKProcessGroup (Private)
ALLOW_DEPRECATED_IMPLEMENTATIONS_END

- (WKContextRef)_contextRef
{
    return toAPI(_processPool.get());
}

#if PLATFORM(IOS_FAMILY)
- (WKGeolocationProviderIOS *)_geolocationProvider
{
    if (!_geolocationProvider)
        _geolocationProvider = adoptNS([[WKGeolocationProviderIOS alloc] initWithProcessPool:*_processPool.get()]);
    return _geolocationProvider.get();
}
#endif // PLATFORM(IOS_FAMILY)

@end
ALLOW_DEPRECATED_DECLARATIONS_END
