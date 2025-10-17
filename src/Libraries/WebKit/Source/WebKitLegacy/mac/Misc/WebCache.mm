/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#import "WebCache.h"

#import "NetworkStorageSessionMap.h"
#import "WebNSObjectExtras.h"
#import "WebPreferences.h"
#import "WebView.h"
#import "WebViewInternal.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/CookieJar.h>
#import <WebCore/CredentialStorage.h>
#import <WebCore/CrossOriginPreflightResultCache.h>
#import <WebCore/Document.h>
#import <WebCore/MemoryCache.h>
#import <WebCore/NetworkStorageSession.h>
#import <WebCore/StorageSessionProvider.h>
#import <WebCore/WebCoreJITOperations.h>
#import <wtf/MainThread.h>
#import <wtf/RunLoop.h>

#if PLATFORM(IOS_FAMILY)
#import "WebFrameInternal.h"
#import <WebCore/BackForwardCache.h>
#import <WebCore/CachedImage.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/WebCoreThreadRun.h>
#endif

class DefaultStorageSessionProvider : public WebCore::StorageSessionProvider {
    WebCore::NetworkStorageSession* storageSession() const final
    {
        return &NetworkStorageSessionMap::defaultStorageSession();
    }
};

@implementation WebCache

+ (void)initialize
{
#if !PLATFORM(IOS_FAMILY)
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
#endif
}

+ (NSArray *)statistics
{
    auto s = WebCore::MemoryCache::singleton().getStatistics();
    return @[
        @{
            @"Images": @(s.images.count),
            @"CSS": @(s.cssStyleSheets.count),
#if ENABLE(XSLT)
            @"XSL": @(s.xslStyleSheets.count),
#else
            @"XSL": @(0),
#endif
            @"JavaScript": @(s.scripts.count),
        },
        @{
            @"Images": @(s.images.size),
            @"CSS": @(s.cssStyleSheets.size),
#if ENABLE(XSLT)
            @"XSL": @(s.xslStyleSheets.size),
#else
            @"XSL": @(0),
#endif
            @"JavaScript": @(s.scripts.size),
        },
        @{
            @"Images": @(s.images.liveSize),
            @"CSS": @(s.cssStyleSheets.liveSize),
#if ENABLE(XSLT)
            @"XSL": @(s.xslStyleSheets.liveSize),
#else
            @"XSL": @(0),
#endif
            @"JavaScript": @(s.scripts.liveSize),
        },
        @{
            @"Images": @(s.images.decodedSize),
            @"CSS": @(s.cssStyleSheets.decodedSize),
#if ENABLE(XSLT)
            @"XSL": @(s.xslStyleSheets.decodedSize),
#else
            @"XSL": @(0),
#endif
            @"JavaScript": @(s.scripts.decodedSize),
        },
    ];
}

+ (void)empty
{
    // Toggling the cache model like this forces the cache to evict all its in-memory resources.
    WebCacheModel cacheModel = [WebView _cacheModel];
    [WebView _setCacheModel:WebCacheModelDocumentViewer];
    [WebView _setCacheModel:cacheModel];

    // Empty the Cross-Origin Preflight cache
    WebCore::CrossOriginPreflightResultCache::singleton().clear();
}

#if PLATFORM(IOS_FAMILY)

+ (void)emptyInMemoryResources
{
    // This method gets called from MobileSafari after it calls [WebView
    // _close]. [WebView _close] schedules its work on the WebThread. So we
    // schedule this method on the WebThread as well so as to pick up all the
    // dead resources left behind after closing the WebViews
    WebThreadRun(^{
        // Toggling the cache model like this forces the cache to evict all its in-memory resources.
        WebCacheModel cacheModel = [WebView _cacheModel];
        [WebView _setCacheModel:WebCacheModelDocumentViewer];
        [WebView _setCacheModel:cacheModel];

        WebCore::MemoryCache::singleton().pruneLiveResources(true);
    });
}

+ (void)sizeOfDeadResources:(int *)resources
{
    WebCore::MemoryCache::Statistics stats = WebCore::MemoryCache::singleton().getStatistics();
    if (resources) {
        *resources = (stats.images.size - stats.images.liveSize)
                     + (stats.cssStyleSheets.size - stats.cssStyleSheets.liveSize)
#if ENABLE(XSLT)
                     + (stats.xslStyleSheets.size - stats.xslStyleSheets.liveSize)
#endif
                     + (stats.scripts.size - stats.scripts.liveSize);
    }
}

+ (CGImageRef)imageForURL:(NSURL *)url
{
    if (!url)
        return nullptr;
    
    WebCore::ResourceRequest request(url);
    WebCore::CachedResource* cachedResource = WebCore::MemoryCache::singleton().resourceForRequest(request, PAL::SessionID::defaultSessionID());
    if (!is<WebCore::CachedImage>(cachedResource))
        return nullptr;

    auto& cachedImage = downcast<WebCore::CachedImage>(*cachedResource);
    if (!cachedImage.hasImage())
        return nullptr;
    
    auto nativeImage = cachedImage.image()->nativeImage();
    if (!nativeImage)
        return nullptr;

    return nativeImage->platformImage().get();
}

#endif // PLATFORM(IOS_FAMILY)

+ (void)setDisabled:(BOOL)disabled
{
    if (!pthread_main_np())
        return [[self _webkit_invokeOnMainThread] setDisabled:disabled];

    WebCore::MemoryCache::singleton().setDisabled(disabled);
}

+ (BOOL)isDisabled
{
    return WebCore::MemoryCache::singleton().disabled();
}

+ (void)clearCachedCredentials
{
    [WebView _makeAllWebViewsPerformSelector:@selector(_clearCredentials)];
    NetworkStorageSessionMap::defaultStorageSession().credentialStorage().clearCredentials();
}

@end
