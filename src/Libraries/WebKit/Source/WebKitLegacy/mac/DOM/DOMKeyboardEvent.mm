/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#import "DOMKeyboardEvent.h"

#import "DOMAbstractViewInternal.h"
#import "DOMEventInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/KeyboardEvent.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::KeyboardEvent*>(reinterpret_cast<WebCore::Event*>(_internal))

@implementation DOMKeyboardEvent

- (NSString *)keyIdentifier
{
    WebCore::JSMainThreadNullState state;
    return IMPL->keyIdentifier();
}

- (unsigned)location
{
    WebCore::JSMainThreadNullState state;
    return IMPL->location();
}

- (unsigned)keyLocation
{
    WebCore::JSMainThreadNullState state;
    return IMPL->location();
}

- (BOOL)ctrlKey
{
    WebCore::JSMainThreadNullState state;
    return IMPL->ctrlKey();
}

- (BOOL)shiftKey
{
    WebCore::JSMainThreadNullState state;
    return IMPL->shiftKey();
}

- (BOOL)altKey
{
    WebCore::JSMainThreadNullState state;
    return IMPL->altKey();
}

- (BOOL)metaKey
{
    WebCore::JSMainThreadNullState state;
    return IMPL->metaKey();
}

- (BOOL)altGraphKey
{
    return false;
}

- (int)keyCode
{
    WebCore::JSMainThreadNullState state;
    return IMPL->keyCode();
}

- (int)charCode
{
    WebCore::JSMainThreadNullState state;
    return IMPL->charCode();
}

- (BOOL)getModifierState:(NSString *)keyIdentifierArg
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getModifierState(keyIdentifierArg);
}

- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)inKeyIdentifier location:(unsigned)inLocation ctrlKey:(BOOL)inCtrlKey altKey:(BOOL)inAltKey shiftKey:(BOOL)inShiftKey metaKey:(BOOL)inMetaKey altGraphKey:(BOOL)inAltGraphKey
{
    WebCore::JSMainThreadNullState state;
    IMPL->initKeyboardEvent(type, canBubble, cancelable, toWindowProxy(view), inKeyIdentifier, inLocation, inCtrlKey, inAltKey, inShiftKey, inMetaKey);
}

- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)inKeyIdentifier location:(unsigned)inLocation ctrlKey:(BOOL)inCtrlKey altKey:(BOOL)inAltKey shiftKey:(BOOL)inShiftKey metaKey:(BOOL)inMetaKey
{
    WebCore::JSMainThreadNullState state;
    IMPL->initKeyboardEvent(type, canBubble, cancelable, toWindowProxy(view), inKeyIdentifier, inLocation, inCtrlKey, inAltKey, inShiftKey, inMetaKey);
}

- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)inKeyIdentifier keyLocation:(unsigned)inKeyLocation ctrlKey:(BOOL)inCtrlKey altKey:(BOOL)inAltKey shiftKey:(BOOL)inShiftKey metaKey:(BOOL)inMetaKey altGraphKey:(BOOL)inAltGraphKey
{
    WebCore::JSMainThreadNullState state;
    IMPL->initKeyboardEvent(type, canBubble, cancelable, toWindowProxy(view), inKeyIdentifier, inKeyLocation, inCtrlKey, inAltKey, inShiftKey, inMetaKey);
}

- (void)initKeyboardEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view keyIdentifier:(NSString *)inKeyIdentifier keyLocation:(unsigned)inKeyLocation ctrlKey:(BOOL)inCtrlKey altKey:(BOOL)inAltKey shiftKey:(BOOL)inShiftKey metaKey:(BOOL)inMetaKey
{
    WebCore::JSMainThreadNullState state;
    IMPL->initKeyboardEvent(type, canBubble, cancelable, toWindowProxy(view), inKeyIdentifier, inKeyLocation, inCtrlKey, inAltKey, inShiftKey, inMetaKey);
}

@end

#undef IMPL
