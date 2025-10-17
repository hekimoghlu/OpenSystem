/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#import "DOMTokenListInternal.h"

#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/DOMTokenList.h>
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::DOMTokenList*>(_internal)

@implementation DOMTokenList

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMTokenList class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (unsigned)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (NSString *)value
{
    WebCore::JSMainThreadNullState state;
    return IMPL->value();
}

- (void)setValue:(NSString *)newValue
{
    WebCore::JSMainThreadNullState state;
    IMPL->setValue(newValue);
}

- (NSString *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return IMPL->item(index);
}

- (BOOL)contains:(NSString *)token
{
    WebCore::JSMainThreadNullState state;
    return IMPL->contains(token);
}

- (BOOL)toggle:(NSString *)token force:(BOOL)force
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->toggle(token, force));
}

@end


DOMTokenList *kit(WebCore::DOMTokenList* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMTokenList *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMTokenList alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
