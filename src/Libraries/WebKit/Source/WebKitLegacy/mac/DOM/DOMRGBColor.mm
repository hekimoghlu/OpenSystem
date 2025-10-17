/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#import "DOMRGBColorInternal.h"

#import "DOMCSSPrimitiveValueInternal.h"
#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ColorCocoa.h>
#import <WebCore/DeprecatedCSSOMPrimitiveValue.h>
#import <WebCore/DeprecatedCSSOMRGBColor.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#if PLATFORM(MAC)
#import <WebCore/ColorMac.h>
#else
#import <WebCore/ColorSpace.h>
#endif

#define IMPL reinterpret_cast<WebCore::DeprecatedCSSOMRGBColor*>(_internal)

@implementation DOMRGBColor

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMRGBColor class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (DOMCSSPrimitiveValue *)red
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->red()));
}

- (DOMCSSPrimitiveValue *)green
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->green()));
}

- (DOMCSSPrimitiveValue *)blue
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->blue()));
}

- (DOMCSSPrimitiveValue *)alpha
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->alpha()));
}

#if PLATFORM(MAC)
- (NSColor *)color
{
    WebCore::JSMainThreadNullState state;
    return cocoaColor(IMPL->color()).autorelease();
}
#else
- (CGColorRef)color
{
    WebCore::JSMainThreadNullState state;
    return WebCore::cachedCGColor(IMPL->color()).autorelease();
}
#endif

@end

DOMRGBColor *kit(WebCore::DeprecatedCSSOMRGBColor* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMRGBColor *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMRGBColor alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
