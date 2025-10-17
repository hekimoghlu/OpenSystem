/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#import "DOMHTMLMarqueeElement.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLMarqueeElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLMarqueeElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLMarqueeElement

- (NSString *)behavior
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::behaviorAttr);
}

- (void)setBehavior:(NSString *)newBehavior
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::behaviorAttr, newBehavior);
}

- (NSString *)bgColor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::bgcolorAttr);
}

- (void)setBgColor:(NSString *)newBgColor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::bgcolorAttr, newBgColor);
}

- (NSString *)direction
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::directionAttr);
}

- (void)setDirection:(NSString *)newDirection
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::directionAttr, newDirection);
}

- (NSString *)height
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::heightAttr);
}

- (void)setHeight:(NSString *)newHeight
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::heightAttr, newHeight);
}

- (unsigned)hspace
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getUnsignedIntegralAttribute(WebCore::HTMLNames::hspaceAttr);
}

- (void)setHspace:(unsigned)newHspace
{
    WebCore::JSMainThreadNullState state;
    IMPL->setUnsignedIntegralAttribute(WebCore::HTMLNames::hspaceAttr, newHspace);
}

- (int)loop
{
    WebCore::JSMainThreadNullState state;
    return IMPL->loop();
}

- (void)setLoop:(int)newLoop
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setLoop(newLoop));
}

- (unsigned)scrollAmount
{
    WebCore::JSMainThreadNullState state;
    return IMPL->scrollAmount();
}

- (void)setScrollAmount:(unsigned)newScrollAmount
{
    WebCore::JSMainThreadNullState state;
    IMPL->setScrollAmount(newScrollAmount);
}

- (unsigned)scrollDelay
{
    WebCore::JSMainThreadNullState state;
    return IMPL->scrollDelay();
}

- (void)setScrollDelay:(unsigned)newScrollDelay
{
    WebCore::JSMainThreadNullState state;
    IMPL->setScrollDelay(newScrollDelay);
}

- (BOOL)trueSpeed
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::truespeedAttr);
}

- (void)setTrueSpeed:(BOOL)newTrueSpeed
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::truespeedAttr, newTrueSpeed);
}

- (unsigned)vspace
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getUnsignedIntegralAttribute(WebCore::HTMLNames::vspaceAttr);
}

- (void)setVspace:(unsigned)newVspace
{
    WebCore::JSMainThreadNullState state;
    IMPL->setUnsignedIntegralAttribute(WebCore::HTMLNames::vspaceAttr, newVspace);
}

- (NSString *)width
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::widthAttr);
}

- (void)setWidth:(NSString *)newWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::widthAttr, newWidth);
}

- (void)start
{
    WebCore::JSMainThreadNullState state;
    IMPL->start();
}

- (void)stop
{
    WebCore::JSMainThreadNullState state;
    IMPL->stop();
}

@end

#undef IMPL
