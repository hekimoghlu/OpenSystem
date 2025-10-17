/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#import "DOMWheelEventInternal.h"

#import "DOMAbstractViewInternal.h"
#import "DOMEventInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/WheelEvent.h>
#import <wtf/GetPtr.h>

#define IMPL static_cast<WebCore::WheelEvent*>(reinterpret_cast<WebCore::Event*>(_internal))

@implementation DOMWheelEvent

- (double)deltaX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->deltaX();
}

- (double)deltaY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->deltaY();
}

- (double)deltaZ
{
    WebCore::JSMainThreadNullState state;
    return IMPL->deltaZ();
}

- (unsigned)deltaMode
{
    WebCore::JSMainThreadNullState state;
    return IMPL->deltaMode();
}

- (int)wheelDeltaX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->wheelDeltaX();
}

- (int)wheelDeltaY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->wheelDeltaY();
}

- (int)wheelDelta
{
    WebCore::JSMainThreadNullState state;
    return IMPL->wheelDelta();
}

- (BOOL)webkitDirectionInvertedFromDevice
{
    WebCore::JSMainThreadNullState state;
    return IMPL->webkitDirectionInvertedFromDevice();
}

- (BOOL)isHorizontal
{
    return !!self.wheelDeltaX;
}

- (void)initWheelEvent:(int)inWheelDeltaX wheelDeltaY:(int)inWheelDeltaY view:(DOMAbstractView *)view screenX:(int)screenX screenY:(int)screenY clientX:(int)clientX clientY:(int)clientY ctrlKey:(BOOL)ctrlKey altKey:(BOOL)altKey shiftKey:(BOOL)shiftKey metaKey:(BOOL)metaKey
{
}

@end

WebCore::WheelEvent* core(DOMWheelEvent *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::WheelEvent*>(wrapper->_internal) : 0;
}

#undef IMPL
