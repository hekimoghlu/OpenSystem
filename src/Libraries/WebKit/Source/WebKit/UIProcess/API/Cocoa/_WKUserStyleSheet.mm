/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#import "_WKUserStyleSheetInternal.h"

#import "APIArray.h"
#import "WKContentWorldInternal.h"
#import "WKNSArray.h"
#import "WKNSURLExtras.h"
#import "WKWebViewInternal.h"
#import "WebKit2Initialize.h"
#import "WebPageProxy.h"
#import "_WKUserContentWorldInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKUserStyleSheet

- (instancetype)initWithSource:(NSString *)source forMainFrameOnly:(BOOL)forMainFrameOnly
{
    if (!(self = [super init]))
        return nil;

    WebKit::InitializeWebKit2();

    API::Object::constructInWrapper<API::UserStyleSheet>(self, WebCore::UserStyleSheet { source, { }, { }, { }, forMainFrameOnly ? WebCore::UserContentInjectedFrames::InjectInTopFrameOnly : WebCore::UserContentInjectedFrames::InjectInAllFrames, WebCore::UserContentMatchParentFrame::Never, WebCore::UserStyleLevel::User }, API::ContentWorld::pageContentWorldSingleton());

    return self;
}

- (instancetype)initWithSource:(NSString *)source forWKWebView:(WKWebView *)webView forMainFrameOnly:(BOOL)forMainFrameOnly includeMatchPatternStrings:(NSArray<NSString *> *)includeMatchPatternStrings excludeMatchPatternStrings:(NSArray<NSString *> *)excludeMatchPatternStrings baseURL:(NSURL *)baseURL level:(_WKUserStyleLevel)level contentWorld:(WKContentWorld *)contentWorld
{

    WebKit::InitializeWebKit2();

    API::Object::constructInWrapper<API::UserStyleSheet>(self, WebCore::UserStyleSheet { source, baseURL, makeVector<String>(includeMatchPatternStrings), makeVector<String>(excludeMatchPatternStrings), forMainFrameOnly ? WebCore::UserContentInjectedFrames::InjectInTopFrameOnly : WebCore::UserContentInjectedFrames::InjectInAllFrames, WebCore::UserContentMatchParentFrame::Never, API::toWebCoreUserStyleLevel(level), webView ? std::optional<WebCore::PageIdentifier>([webView _page]->webPageIDInMainFrameProcess()) : std::nullopt }, contentWorld ? *contentWorld->_contentWorld : API::ContentWorld::pageContentWorldSingleton());

    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKUserStyleSheet.class, self))
        return;

    _userStyleSheet->~UserStyleSheet();

    [super dealloc];
}

- (NSString *)source
{
    return _userStyleSheet->userStyleSheet().source();
}

- (NSURL *)baseURL
{
    return _userStyleSheet->userStyleSheet().url();
}

- (BOOL)isForMainFrameOnly
{
    return _userStyleSheet->userStyleSheet().injectedFrames() == WebCore::UserContentInjectedFrames::InjectInTopFrameOnly;
}

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_userStyleSheet;
}

@end
