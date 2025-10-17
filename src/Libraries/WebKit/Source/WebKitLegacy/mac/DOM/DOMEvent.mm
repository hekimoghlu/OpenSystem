/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#import "DOMEventInternal.h"

#import "DOMEventTarget.h"
#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/Event.h>
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/Node.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::Event*>(_internal)

@implementation DOMEvent

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMEvent class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->type();
}

- (id <DOMEventTarget>)target
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->target()));
}

- (id <DOMEventTarget>)currentTarget
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->currentTarget()));
}

- (unsigned short)eventPhase
{
    WebCore::JSMainThreadNullState state;
    return IMPL->eventPhase();
}

- (BOOL)bubbles
{
    WebCore::JSMainThreadNullState state;
    return IMPL->bubbles();
}

- (BOOL)cancelable
{
    WebCore::JSMainThreadNullState state;
    return IMPL->cancelable();
}

- (BOOL)composed
{
    WebCore::JSMainThreadNullState state;
    return IMPL->composed();
}

- (DOMTimeStamp)timeStamp
{
    WebCore::JSMainThreadNullState state;
    return IMPL->timeStamp().approximateWallTime().secondsSinceEpoch().milliseconds();
}

- (BOOL)defaultPrevented
{
    WebCore::JSMainThreadNullState state;
    return IMPL->defaultPrevented();
}

- (BOOL)isTrusted
{
    WebCore::JSMainThreadNullState state;
    return IMPL->isTrusted();
}

- (id <DOMEventTarget>)srcElement
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->target()));
}

- (BOOL)returnValue
{
    WebCore::JSMainThreadNullState state;
    return IMPL->legacyReturnValue();
}

- (void)setReturnValue:(BOOL)newReturnValue
{
    WebCore::JSMainThreadNullState state;
    IMPL->setLegacyReturnValue(newReturnValue);
}

- (BOOL)cancelBubble
{
    WebCore::JSMainThreadNullState state;
    return IMPL->cancelBubble();
}

- (void)setCancelBubble:(BOOL)newCancelBubble
{
    WebCore::JSMainThreadNullState state;
    IMPL->setCancelBubble(newCancelBubble);
}

- (void)stopPropagation
{
    WebCore::JSMainThreadNullState state;
    IMPL->stopPropagation();
}

- (void)preventDefault
{
    WebCore::JSMainThreadNullState state;
    IMPL->preventDefault();
}

- (void)initEvent:(NSString *)eventTypeArg canBubbleArg:(BOOL)canBubbleArg cancelableArg:(BOOL)cancelableArg
{
    WebCore::JSMainThreadNullState state;
    IMPL->initEvent(eventTypeArg, canBubbleArg, cancelableArg);
}

- (void)stopImmediatePropagation
{
    WebCore::JSMainThreadNullState state;
    IMPL->stopImmediatePropagation();
}

@end

@implementation DOMEvent (DOMEventDeprecated)

- (void)initEvent:(NSString *)eventTypeArg :(BOOL)canBubbleArg :(BOOL)cancelableArg
{
    [self initEvent:eventTypeArg canBubbleArg:canBubbleArg cancelableArg:cancelableArg];
}

@end

WebCore::Event* core(DOMEvent *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::Event*>(wrapper->_internal) : 0;
}

DOMEvent *kit(WebCore::Event* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMEvent *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    RetainPtr<DOMEvent> wrapper = adoptNS([[kitClass(value) alloc] _init]);
    if (!wrapper)
        return nil;
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
