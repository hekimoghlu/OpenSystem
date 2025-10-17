/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#import "WKWebProcessPlugInCSSStyleDeclarationHandleInternal.h"

#import <WebCore/CSSStyleDeclaration.h>
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKWebProcessPlugInCSSStyleDeclarationHandle {
    API::ObjectStorage<WebKit::InjectedBundleCSSStyleDeclarationHandle> _cssStyleDeclarationHandle;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebProcessPlugInCSSStyleDeclarationHandle.class, self))
        return;
    _cssStyleDeclarationHandle->~InjectedBundleCSSStyleDeclarationHandle();
    [super dealloc];
}

+ (WKWebProcessPlugInCSSStyleDeclarationHandle *)cssStyleDeclarationHandleWithJSValue:(JSValue *)value inContext:(JSContext *)context
{
    JSContextRef contextRef = [context JSGlobalContextRef];
    JSObjectRef objectRef = JSValueToObject(contextRef, [value JSValueRef], nullptr);
    return wrapper(WebKit::InjectedBundleCSSStyleDeclarationHandle::getOrCreate(contextRef, objectRef)).autorelease();
}

- (WebKit::InjectedBundleCSSStyleDeclarationHandle&)_cssStyleDeclarationHandle
{
    return *_cssStyleDeclarationHandle;
}

// MARK: WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_cssStyleDeclarationHandle;
}

@end
