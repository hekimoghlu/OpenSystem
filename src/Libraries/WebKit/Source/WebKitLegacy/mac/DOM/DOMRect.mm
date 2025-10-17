/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#import "DOMRectInternal.h"

#import "DOMCSSPrimitiveValueInternal.h"
#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/DeprecatedCSSOMPrimitiveValue.h>
#import <WebCore/DeprecatedCSSOMRect.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#define IMPL reinterpret_cast<WebCore::DeprecatedCSSOMRect*>(_internal)

@implementation DOMRect

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMRect class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (DOMCSSPrimitiveValue *)top
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->top()));
}

- (DOMCSSPrimitiveValue *)right
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->right()));
}

- (DOMCSSPrimitiveValue *)bottom
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->bottom()));
}

- (DOMCSSPrimitiveValue *)left
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->left()));
}

@end

DOMRect *kit(WebCore::DeprecatedCSSOMRect* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMRect *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMRect alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
