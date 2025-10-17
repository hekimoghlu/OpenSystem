/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#import "DOMHTMLScriptElementInternal.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLScriptElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLScriptElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLScriptElement

- (NSString *)text
{
    WebCore::JSMainThreadNullState state;
    return IMPL->text();
}

- (void)setText:(NSString *)newText
{
    WebCore::JSMainThreadNullState state;
    IMPL->setText(String(newText));
}

- (NSString *)htmlFor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::forAttr);
}

- (void)setHtmlFor:(NSString *)newHtmlFor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::forAttr, newHtmlFor);
}

- (NSString *)event
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::eventAttr);
}

- (void)setEvent:(NSString *)newEvent
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::eventAttr, newEvent);
}

- (NSString *)charset
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::charsetAttr);
}

- (void)setCharset:(NSString *)newCharset
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::charsetAttr, newCharset);
}

- (BOOL)async
{
    WebCore::JSMainThreadNullState state;
    return IMPL->async();
}

- (void)setAsync:(BOOL)newAsync
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAsync(newAsync);
}

- (BOOL)defer
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::deferAttr);
}

- (void)setDefer:(BOOL)newDefer
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::deferAttr, newDefer);
}

- (NSString *)src
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::srcAttr).string();
}

- (void)setSrc:(NSString *)newSrc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::srcAttr, newSrc);
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::typeAttr);
}

- (void)setType:(NSString *)newType
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::typeAttr, newType);
}

- (NSString *)crossOrigin
{
    WebCore::JSMainThreadNullState state;
    return IMPL->crossOrigin();
}

- (void)setCrossOrigin:(NSString *)newCrossOrigin
{
    WebCore::JSMainThreadNullState state;
    IMPL->setCrossOrigin(newCrossOrigin);
}

- (NSString *)nonce
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::nonceAttr);
}

- (void)setNonce:(NSString *)newNonce
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::nonceAttr, newNonce);
}

@end

DOMHTMLScriptElement *kit(WebCore::HTMLScriptElement* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLScriptElement*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
