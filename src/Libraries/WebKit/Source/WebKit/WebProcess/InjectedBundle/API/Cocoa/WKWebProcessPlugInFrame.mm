/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#import "WKWebProcessPlugInFrameInternal.h"

#import "WKNSArray.h"
#import "WKNSURLExtras.h"
#import "WKWebProcessPlugInBrowserContextControllerInternal.h"
#import "WKWebProcessPlugInCSSStyleDeclarationHandleInternal.h"
#import "WKWebProcessPlugInHitTestResultInternal.h"
#import "WKWebProcessPlugInNodeHandleInternal.h"
#import "WKWebProcessPlugInRangeHandleInternal.h"
#import "WKWebProcessPlugInScriptWorldInternal.h"
#import "WebProcess.h"
#import "_WKFrameHandleInternal.h"
#import <JavaScriptCore/JSValue.h>
#import <WebCore/CertificateInfo.h>
#import <WebCore/IntPoint.h>
#import <WebCore/LinkIconCollector.h>
#import <WebCore/LinkIconType.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation WKWebProcessPlugInFrame {
    API::ObjectStorage<WebKit::WebFrame> _frame;
}

+ (instancetype)lookUpFrameFromHandle:(_WKFrameHandle *)handle
{
    auto frameID = handle->_frameHandle->frameID();
    return wrapper(frameID ? WebKit::WebProcess::singleton().webFrame(*frameID) : nullptr);
}

+ (instancetype)lookUpFrameFromJSContext:(JSContext *)context
{
    return wrapper(WebKit::WebFrame::frameForContext(context.JSGlobalContextRef)).autorelease();
}

+ (instancetype)lookUpContentFrameFromWindowOrFrameElement:(JSValue *)value
{
    return wrapper(WebKit::WebFrame::contentFrameForWindowOrFrameElement(value.context.JSGlobalContextRef, value.JSValueRef)).autorelease();
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebProcessPlugInFrame.class, self))
        return;
    _frame->~WebFrame();
    [super dealloc];
}

- (JSContext *)jsContextForWorld:(WKWebProcessPlugInScriptWorld *)world
{
    return [JSContext contextWithJSGlobalContextRef:_frame->jsContextForWorld(&[world _scriptWorld])];
}

- (JSContext *)jsContextForServiceWorkerWorld:(WKWebProcessPlugInScriptWorld *)world
{
    if (auto context = _frame->jsContextForServiceWorkerWorld(&[world _scriptWorld]))
        return [JSContext contextWithJSGlobalContextRef:context];
    return nil;
}

- (WKWebProcessPlugInHitTestResult *)hitTest:(CGPoint)point
{
    return wrapper(_frame->hitTest(WebCore::IntPoint(point))).autorelease();
}

- (WKWebProcessPlugInHitTestResult *)hitTest:(CGPoint)point options:(WKHitTestOptions)options
{
    auto types = WebKit::WebFrame::defaultHitTestRequestTypes();
    if (options & WKHitTestOptionAllowUserAgentShadowRootContent)
        types.remove(WebCore::HitTestRequest::Type::DisallowUserAgentShadowContent);
    return wrapper(_frame->hitTest(WebCore::IntPoint(point), types)).autorelease();
}

- (JSValue *)jsCSSStyleDeclarationForCSSStyleDeclarationHandle:(WKWebProcessPlugInCSSStyleDeclarationHandle *)cssStyleDeclarationHandle inWorld:(WKWebProcessPlugInScriptWorld *)world
{
    JSValueRef valueRef = _frame->jsWrapperForWorld(&[cssStyleDeclarationHandle _cssStyleDeclarationHandle], &[world _scriptWorld]);
    return [JSValue valueWithJSValueRef:valueRef inContext:[self jsContextForWorld:world]];
}

- (JSValue *)jsNodeForNodeHandle:(WKWebProcessPlugInNodeHandle *)nodeHandle inWorld:(WKWebProcessPlugInScriptWorld *)world
{
    JSValueRef valueRef = _frame->jsWrapperForWorld(&[nodeHandle _nodeHandle], &[world _scriptWorld]);
    return [JSValue valueWithJSValueRef:valueRef inContext:[self jsContextForWorld:world]];
}

- (JSValue *)jsRangeForRangeHandle:(WKWebProcessPlugInRangeHandle *)rangeHandle inWorld:(WKWebProcessPlugInScriptWorld *)world
{
    JSValueRef valueRef = _frame->jsWrapperForWorld(&[rangeHandle _rangeHandle], &[world _scriptWorld]);
    return [JSValue valueWithJSValueRef:valueRef inContext:[self jsContextForWorld:world]];
}

- (WKWebProcessPlugInBrowserContextController *)_browserContextController
{
    if (!_frame->page())
        return nil;
    return WebKit::wrapper(*_frame->page());
}

- (NSURL *)URL
{
    return _frame->url();
}

- (NSArray *)childFrames
{
    return WebKit::wrapper(_frame->childFrames()).autorelease();
}

- (BOOL)containsAnyFormElements
{
    return !!_frame->containsAnyFormElements();
}

- (BOOL)isMainFrame
{
    return !!_frame->isMainFrame();
}

- (_WKFrameHandle *)handle
{
    return wrapper(API::FrameHandle::create(_frame->frameID())).autorelease();
}

- (NSString *)_securityOrigin
{
    auto* coreFrame = _frame->coreLocalFrame();
    if (!coreFrame)
        return nil;
    return coreFrame->document()->securityOrigin().toString();
}

static RetainPtr<NSArray> collectIcons(WebCore::LocalFrame* frame, OptionSet<WebCore::LinkIconType> iconTypes)
{
    if (!frame)
        return @[];
    RefPtr document = frame->document();
    if (!document)
        return @[];
    return createNSArray(WebCore::LinkIconCollector(*document).iconsOfTypes(iconTypes), [] (auto&& icon) -> NSURL * {
        return icon.url;
    });
}

- (NSArray *)appleTouchIconURLs
{
    return collectIcons(_frame->coreLocalFrame(), { WebCore::LinkIconType::TouchIcon, WebCore::LinkIconType::TouchPrecomposedIcon }).autorelease();
}

- (NSArray *)faviconURLs
{
    return collectIcons(_frame->coreLocalFrame(), WebCore::LinkIconType::Favicon).autorelease();
}

- (WKWebProcessPlugInFrame *)_parentFrame
{
    return wrapper(_frame->parentFrame()).autorelease();
}

- (BOOL)_hasCustomContentProvider
{
    if (!_frame->isMainFrame())
        return false;

    return _frame->page()->mainFrameHasCustomContentProvider();
}

- (NSArray *)_certificateChain
{
    return (NSArray *)WebCore::CertificateInfo::certificateChainFromSecTrust(_frame->certificateInfo().trust().get()).autorelease();
}

- (SecTrustRef)_serverTrust
{
    return _frame->certificateInfo().trust().get();
}

- (NSURL *)_provisionalURL
{
    return [NSURL _web_URLWithWTFString:_frame->provisionalURL()];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_frame;
}

@end
