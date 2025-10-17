/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#import "WKScriptMessageInternal.h"

#import <WebKit/WKFrameInfo.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

@implementation WKScriptMessage {
    RetainPtr<id> _body;
    WeakObjCPtr<WKWebView> _webView;
    RetainPtr<WKFrameInfo> _frameInfo;
    RetainPtr<NSString> _name;
    RetainPtr<WKContentWorld> _world;
}

- (instancetype)_initWithBody:(id)body webView:(WKWebView *)webView frameInfo:(WKFrameInfo *)frameInfo name:(NSString *)name world:(WKContentWorld *)world
{
    if (!(self = [super init]))
        return nil;

    _body = adoptNS([body copy]);
    _webView = webView;
    _frameInfo = frameInfo;
    _name = adoptNS([name copy]);
    _world = world;

    return self;
}

- (id)body
{
    return _body.get();
}

- (WKWebView *)webView
{
    return _webView.getAutoreleased();
}

- (WKFrameInfo *)frameInfo
{
    return _frameInfo.get();
}

- (NSString *)name
{
    return _name.get();
}

- (WKContentWorld *)world
{
    return _world.get();
}

@end
