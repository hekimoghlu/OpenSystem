/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#import "DOMHTMLFormElementInternal.h"

#import "DOMHTMLCollectionInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLCollection.h>
#import <WebCore/HTMLFormElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLFormElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLFormElement

- (NSString *)acceptCharset
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::accept_charsetAttr);
}

- (void)setAcceptCharset:(NSString *)newAcceptCharset
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::accept_charsetAttr, newAcceptCharset);
}

- (NSString *)action
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::actionAttr).string();
}

- (void)setAction:(NSString *)newAction
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::actionAttr, newAction);
}

- (NSString *)autocomplete
{
    WebCore::JSMainThreadNullState state;
    return IMPL->autocomplete();
}

- (void)setAutocomplete:(NSString *)newAutocomplete
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAutocomplete(newAutocomplete);
}

- (NSString *)enctype
{
    WebCore::JSMainThreadNullState state;
    return IMPL->enctype();
}

- (void)setEnctype:(NSString *)newEnctype
{
    WebCore::JSMainThreadNullState state;
    IMPL->setEnctype(newEnctype);
}

- (NSString *)encoding
{
    WebCore::JSMainThreadNullState state;
    return IMPL->enctype();
}

- (void)setEncoding:(NSString *)newEncoding
{
    WebCore::JSMainThreadNullState state;
    IMPL->setEnctype(newEncoding);
}

- (NSString *)method
{
    WebCore::JSMainThreadNullState state;
    return IMPL->method();
}

- (void)setMethod:(NSString *)newMethod
{
    WebCore::JSMainThreadNullState state;
    IMPL->setMethod(newMethod);
}

- (NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getNameAttribute();
}

- (void)setName:(NSString *)newName
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::nameAttr, newName);
}

- (BOOL)noValidate
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::novalidateAttr);
}

- (void)setNoValidate:(BOOL)newNoValidate
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::novalidateAttr, newNoValidate);
}

- (NSString *)target
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::targetAttr);
}

- (void)setTarget:(NSString *)newTarget
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::targetAttr, newTarget);
}

- (DOMHTMLCollection *)elements
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->elementsForNativeBindings()));
}

- (int)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (void)submit
{
    WebCore::JSMainThreadNullState state;
    IMPL->submit();
}

- (void)reset
{
    WebCore::JSMainThreadNullState state;
    IMPL->reset();
}

- (BOOL)checkValidity
{
    WebCore::JSMainThreadNullState state;
    return IMPL->checkValidity();
}

@end

DOMHTMLFormElement *kit(WebCore::HTMLFormElement* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLFormElement*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
