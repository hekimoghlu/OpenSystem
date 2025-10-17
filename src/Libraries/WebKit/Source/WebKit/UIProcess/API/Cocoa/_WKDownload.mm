/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#import "_WKDownloadInternal.h"

#import "APIDownloadClient.h"
#import "DownloadProxy.h"
#import "WKDownloadInternal.h"
#import "WKFrameInfoInternal.h"
#import "WKNSData.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <wtf/WeakObjCPtr.h>
#import <wtf/cocoa/VectorCocoa.h>


ALLOW_DEPRECATED_DECLARATIONS_BEGIN
static NSMapTable<WKDownload *, _WKDownload *> *downloadWrapperMap()
{
    static NeverDestroyed<RetainPtr<NSMapTable>> table;
    if (!table.get())
        table.get() = [NSMapTable weakToWeakObjectsMapTable];
    return table.get().get();
}
ALLOW_DEPRECATED_DECLARATIONS_END

// FIXME: Remove when rdar://133558571, rdar://133558520, rdar://133498655, rdar://133498564, rdar://133498491, rdar://133495572, and rdar://125569813 are complete.

IGNORE_WARNINGS_BEGIN("deprecated-implementations")
@implementation _WKDownload
IGNORE_WARNINGS_END

- (instancetype)initWithDownload2:(WKDownload *)download
{
    if (!(self = [super init]))
        return nil;
    _download = download;
    return self;
}

+ (instancetype)downloadWithDownload:(WKDownload *)download
{
    if (_WKDownload *wrapper = [downloadWrapperMap() objectForKey:download])
        return wrapper;
    auto wrapper = adoptNS([[_WKDownload alloc] initWithDownload2:download]);
    [downloadWrapperMap() setObject:wrapper.get() forKey:download];
    return wrapper.autorelease();
}

- (void)cancel
{
    _download->_download->cancel([download = Ref { *_download->_download }] (auto*) {
        download->client().legacyDidCancel(download.get());
    });
}

- (void)publishProgressAtURL:(NSURL *)URL
{
    _download->_download->publishProgress(URL);
}

- (NSURLRequest *)request
{
    return _download->_download->request().nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
}

- (WKWebView *)originatingWebView
{
    auto page = _download->_download->originatingPage();
    return page ? page->cocoaView().autorelease() : nil;
}

-(NSArray<NSURL *> *)redirectChain
{
    return createNSArray(_download->_download->redirectChain(), [] (auto& url) -> NSURL * {
        return url;
    }).autorelease();
}

- (BOOL)wasUserInitiated
{
    return _download->_download->wasUserInitiated();
}

- (NSData *)resumeData
{
    return WebKit::wrapper(_download->_download->legacyResumeData());
}

- (WKFrameInfo *)originatingFrame
{
    return WebKit::wrapper(&_download->_download->frameInfo());
}

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_download->_download;
}

@end
