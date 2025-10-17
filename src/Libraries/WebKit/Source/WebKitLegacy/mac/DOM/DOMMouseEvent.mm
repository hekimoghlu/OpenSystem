/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#import "DOMMouseEvent.h"

#import "DOMAbstractViewInternal.h"
#import "DOMEventInternal.h"
#import "DOMEventTarget.h"
#import "DOMNode.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/MouseEvent.h>
#import <WebCore/Node.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::MouseEvent*>(reinterpret_cast<WebCore::Event*>(_internal))

@implementation DOMMouseEvent

- (int)screenX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->screenX();
}

- (int)screenY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->screenY();
}

- (int)clientX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->clientX();
}

- (int)clientY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->clientY();
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

- (short)button
{
    WebCore::JSMainThreadNullState state;
    return IMPL->buttonAsShort();
}

- (id <DOMEventTarget>)relatedTarget
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->relatedTarget()));
}

- (int)offsetX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->offsetX();
}

- (int)offsetY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->offsetY();
}

- (int)x
{
    WebCore::JSMainThreadNullState state;
    return IMPL->clientX();
}

- (int)y
{
    WebCore::JSMainThreadNullState state;
    return IMPL->clientY();
}

- (DOMNode *)fromElement
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->fromElement()));
}

- (DOMNode *)toElement
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->toElement()));
}

- (void)initMouseEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)view detail:(int)detail screenX:(int)inScreenX screenY:(int)inScreenY clientX:(int)inClientX clientY:(int)inClientY ctrlKey:(BOOL)inCtrlKey altKey:(BOOL)inAltKey shiftKey:(BOOL)inShiftKey metaKey:(BOOL)inMetaKey button:(unsigned short)inButton relatedTarget:(id <DOMEventTarget>)inRelatedTarget
{
    WebCore::JSMainThreadNullState state;
    DOMNode* inRelatedTargetObjC = inRelatedTarget;
    WebCore::Node* inRelatedTargetNode = core(inRelatedTargetObjC);
    IMPL->initMouseEvent(type, canBubble, cancelable, toWindowProxy(view), detail, inScreenX, inScreenY, inClientX, inClientY, inCtrlKey, inAltKey, inShiftKey, inMetaKey, inButton, inRelatedTargetNode);
}

@end

@implementation DOMMouseEvent (DOMMouseEventDeprecated)

- (void)initMouseEvent:(NSString *)type :(BOOL)canBubble :(BOOL)cancelable :(DOMAbstractView *)view :(int)detail :(int)inScreenX :(int)inScreenY :(int)inClientX :(int)inClientY :(BOOL)inCtrlKey :(BOOL)inAltKey :(BOOL)inShiftKey :(BOOL)inMetaKey :(unsigned short)inButton :(id <DOMEventTarget>)inRelatedTarget
{
    [self initMouseEvent:type canBubble:canBubble cancelable:cancelable view:view detail:detail screenX:inScreenX screenY:inScreenY clientX:inClientX clientY:inClientY ctrlKey:inCtrlKey altKey:inAltKey shiftKey:inShiftKey metaKey:inMetaKey button:inButton relatedTarget:inRelatedTarget];
}

@end

#undef IMPL
