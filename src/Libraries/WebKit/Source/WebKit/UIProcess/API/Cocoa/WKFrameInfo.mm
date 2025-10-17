/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#import "WKFrameInfoInternal.h"

#import "WKSecurityOriginInternal.h"
#import "WKWebViewInternal.h"
#import "WebFrameProxy.h"
#import "WebPageProxy.h"
#import "_WKFrameHandleInternal.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKFrameInfo

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKFrameInfo.class, self))
        return;

    _frameInfo->~FrameInfo();

    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; webView = %p; isMainFrame = %s; request = %@>", NSStringFromClass(self.class), self, self.webView, self.mainFrame ? "YES" : "NO", self.request];
}

- (BOOL)isMainFrame
{
    return _frameInfo->isMainFrame();
}

- (NSURLRequest *)request
{
    return _frameInfo->request().nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody) ?: [NSURLRequest requestWithURL:adoptNS([[NSURL alloc] initWithString:@""]).get()];
}

- (WKSecurityOrigin *)securityOrigin
{
    auto& data = _frameInfo->securityOrigin();
    auto apiOrigin = API::SecurityOrigin::create(data);
    return retainPtr(wrapper(apiOrigin.get())).autorelease();
}

- (WKWebView *)webView
{
    auto page = _frameInfo->page();
    return page ? page->cocoaView().autorelease() : nil;
}

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_frameInfo;
}

@end

@implementation WKFrameInfo (WKPrivate)

- (_WKFrameHandle *)_handle
{
    return wrapper(_frameInfo->handle()).autorelease();
}

- (_WKFrameHandle *)_parentFrameHandle
{
    return wrapper(_frameInfo->parentFrameHandle()).autorelease();
}

- (NSUUID *)_documentIdentifier
{
    return _frameInfo->documentID()->object();
}

- (pid_t)_processIdentifier
{
    return _frameInfo->processID();
}

- (BOOL)_isLocalFrame
{
    return _frameInfo->isLocalFrame();
}

- (BOOL)_isFocused
{
    return _frameInfo->isFocused();
}

- (BOOL)_errorOccurred
{
    return _frameInfo->errorOccurred();
}

- (NSString *)_title
{
    return _frameInfo->title();
}

- (BOOL)_isScrollable
{
    return _frameInfo->frameInfoData().frameMetrics.isScrollable == WebKit::IsScrollable::Yes;
}

- (CGSize)_contentSize
{
    return (CGSize)_frameInfo->frameInfoData().frameMetrics.contentSize;
}

- (CGSize)_visibleContentSize
{
    return (CGSize)_frameInfo->frameInfoData().frameMetrics.visibleContentSize;
}

- (CGSize)_visibleContentSizeExcludingScrollbars
{
    return (CGSize)_frameInfo->frameInfoData().frameMetrics.visibleContentSizeExcludingScrollbars;
}

@end
