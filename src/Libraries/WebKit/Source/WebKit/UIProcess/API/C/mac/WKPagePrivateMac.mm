/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#import "WKPagePrivateMac.h"

#import "APIPageConfiguration.h"
#import "FullscreenClient.h"
#import "PageLoadStateObserver.h"
#import "WKAPICast.h"
#import "WKNSURLExtras.h"
#import "WKNavigationInternal.h"
#import "WKViewInternal.h"
#import "WKWebViewInternal.h"
#import "WebPageGroup.h"
#import "WebPageProxy.h"
#import "WebPreferences.h"
#import "WebProcessPool.h"
#import <wtf/MainThread.h>

@interface WKObservablePageState : NSObject <_WKObservablePageState> {
    RefPtr<WebKit::WebPageProxy> _page;
    std::unique_ptr<WebKit::PageLoadStateObserver> _observer;
}

@end

@implementation WKObservablePageState

- (id)initWithPage:(RefPtr<WebKit::WebPageProxy>&&)page
{
    if (!(self = [super init]))
        return nil;

    _page = WTFMove(page);
    _observer = makeUniqueWithoutRefCountedCheck<WebKit::PageLoadStateObserver>(self, @"URL");
    _page->pageLoadState().addObserver(*_observer);

    return self;
}

- (void)dealloc
{
    _observer->clearObject();

    ensureOnMainRunLoop([page = WTFMove(_page), observer = WTFMove(_observer)] {
        page->pageLoadState().removeObserver(*observer);
    });

    [super dealloc];
}

- (BOOL)isLoading
{
    return _page->pageLoadState().isLoading();
}

- (NSString *)title
{
    return _page->pageLoadState().title();
}

- (NSURL *)URL
{
    return [NSURL _web_URLWithWTFString:_page->pageLoadState().activeURL()];
}

- (BOOL)hasOnlySecureContent
{
    return _page->pageLoadState().hasOnlySecureContent();
}

- (BOOL)_webProcessIsResponsive
{
    return _page->legacyMainFrameProcess().isResponsive();
}

- (double)estimatedProgress
{
    return _page->estimatedProgress();
}

- (NSURL *)unreachableURL
{
    return [NSURL _web_URLWithWTFString:_page->pageLoadState().unreachableURL()];
}

- (SecTrustRef)serverTrust
{
    return _page->pageLoadState().certificateInfo().trust().get();
}

@end

id <_WKObservablePageState> WKPageCreateObservableState(WKPageRef pageRef)
{
    return [[WKObservablePageState alloc] initWithPage:WebKit::toImpl(pageRef)];
}

_WKRemoteObjectRegistry *WKPageGetObjectRegistry(WKPageRef pageRef)
{
#if PLATFORM(MAC)
    return WebKit::toImpl(pageRef)->remoteObjectRegistry();
#else
    return nil;
#endif
}

bool WKPageIsURLKnownHSTSHost(WKPageRef page, WKURLRef url)
{
    WebKit::WebPageProxy* webPageProxy = WebKit::toImpl(page);

    return webPageProxy->configuration().processPool().isURLKnownHSTSHost(WebKit::toImpl(url)->string());
}

WKNavigation *WKPageLoadURLRequestReturningNavigation(WKPageRef pageRef, WKURLRequestRef urlRequestRef)
{
    auto resourceRequest = WebKit::toImpl(urlRequestRef)->resourceRequest();
    return WebKit::wrapper(WebKit::toImpl(pageRef)->loadRequest(WTFMove(resourceRequest))).autorelease();
}

WKNavigation *WKPageLoadFileReturningNavigation(WKPageRef pageRef, WKURLRef fileURL, WKURLRef resourceDirectoryURL)
{
    return WebKit::wrapper(WebKit::toImpl(pageRef)->loadFile(WebKit::toWTFString(fileURL), WebKit::toWTFString(resourceDirectoryURL))).autorelease();
}

WKWebView *WKPageGetWebView(WKPageRef page)
{
    return page ? WebKit::toImpl(page)->cocoaView().autorelease() : nil;
}

#if PLATFORM(MAC)
bool WKPageIsPlayingVideoInEnhancedFullscreen(WKPageRef pageRef)
{
    return WebKit::toImpl(pageRef)->isPlayingVideoInEnhancedFullscreen();
}
#endif

void WKPageSetFullscreenDelegate(WKPageRef page, id <_WKFullscreenDelegate> delegate)
{
#if ENABLE(FULLSCREEN_API)
    downcast<WebKit::FullscreenClient>(WebKit::toImpl(page)->fullscreenClient()).setDelegate(delegate);
#endif
}

id <_WKFullscreenDelegate> WKPageGetFullscreenDelegate(WKPageRef page)
{
#if ENABLE(FULLSCREEN_API)
    return downcast<WebKit::FullscreenClient>(WebKit::toImpl(page)->fullscreenClient()).delegate().autorelease();
#else
    return nil;
#endif
}

