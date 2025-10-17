/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
#import "WKReloadFrameErrorRecoveryAttempter.h"

#import "_WKErrorRecoveryAttempting.h"
#import "_WKFrameHandleInternal.h"
#import "WKWebViewInternal.h"
#import "WebFrameProxy.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import <WebCore/FrameIdentifier.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

@interface WKReloadFrameErrorRecoveryAttempter ()
@end

@implementation WKReloadFrameErrorRecoveryAttempter {
    WeakObjCPtr<WKWebView> _webView;
    RetainPtr<_WKFrameHandle> _frameHandle;
    String _urlString;
}

- (id)initWithWebView:(WKWebView *)webView frameHandle:(_WKFrameHandle *)frameHandle urlString:(const String&)urlString
{
    if (!(self = [super init]))
        return nil;

    _webView = webView;
    _frameHandle = frameHandle;
    _urlString = urlString;

    return self;
}

- (BOOL)attemptRecovery
{
    auto webView = _webView.get();
    if (!webView)
        return NO;

    RefPtr webFrameProxy = WebKit::WebFrameProxy::webFrame(_frameHandle->_frameHandle->frameID());
    if (!webFrameProxy)
        return NO;

    webFrameProxy->loadURL(URL { _urlString });
    return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    self = [super init];
    return self;
}

+ (BOOL)supportsSecureCoding
{
    return YES;
}

@end
