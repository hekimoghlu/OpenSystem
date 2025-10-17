/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#import "DOMUIEvent.h"

#import "DOMAbstractViewInternal.h"
#import "DOMEventInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/KeyboardEvent.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/UIEvent.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::UIEvent*>(reinterpret_cast<WebCore::Event*>(_internal))

@implementation DOMUIEvent

- (DOMAbstractView *)view
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->view()));
}

- (int)detail
{
    WebCore::JSMainThreadNullState state;
    return IMPL->detail();
}

- (int)keyCode
{
    WebCore::JSMainThreadNullState state;
    if (is<WebCore::KeyboardEvent>(*IMPL))
        return downcast<WebCore::KeyboardEvent>(*IMPL).keyCode();
    return 0;
}

- (int)charCode
{
    WebCore::JSMainThreadNullState state;
    if (is<WebCore::KeyboardEvent>(*IMPL))
        return downcast<WebCore::KeyboardEvent>(*IMPL).charCode();
    return 0;
}

- (int)layerX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->layerX();
}

- (int)layerY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->layerY();
}

- (int)pageX
{
    WebCore::JSMainThreadNullState state;
    return IMPL->pageX();
}

- (int)pageY
{
    WebCore::JSMainThreadNullState state;
    return IMPL->pageY();
}

- (int)which
{
    WebCore::JSMainThreadNullState state;
    return IMPL->which();
}

- (void)initUIEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable view:(DOMAbstractView *)inView detail:(int)inDetail
{
    WebCore::JSMainThreadNullState state;
    IMPL->initUIEvent(type, canBubble, cancelable, toWindowProxy(inView), inDetail);
}

@end

@implementation DOMUIEvent (DOMUIEventDeprecated)

- (void)initUIEvent:(NSString *)type :(BOOL)canBubble :(BOOL)cancelable :(DOMAbstractView *)inView :(int)inDetail
{
    [self initUIEvent:type canBubble:canBubble cancelable:cancelable view:inView detail:inDetail];
}

@end

#undef IMPL
