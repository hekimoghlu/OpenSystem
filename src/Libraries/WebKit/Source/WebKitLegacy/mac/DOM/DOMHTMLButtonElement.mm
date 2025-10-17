/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#import "DOMHTMLButtonElement.h"

#import "DOMHTMLFormElementInternal.h"
#import "DOMNodeInternal.h"
#import "DOMNodeListInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLButtonElement.h>
#import <WebCore/HTMLFormElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/NodeList.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLButtonElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLButtonElement

- (BOOL)autofocus
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::autofocusAttr);
}

- (void)setAutofocus:(BOOL)newAutofocus
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::autofocusAttr, newAutofocus);
}

- (BOOL)disabled
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::disabledAttr);
}

- (void)setDisabled:(BOOL)newDisabled
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::disabledAttr, newDisabled);
}

- (DOMHTMLFormElement *)form
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->form()));
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->type();
}

- (void)setType:(NSString *)newType
{
    WebCore::JSMainThreadNullState state;
    IMPL->setType(newType);
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

- (NSString *)value
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::valueAttr);
}

- (void)setValue:(NSString *)newValue
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::valueAttr, newValue);
}

- (BOOL)willValidate
{
    WebCore::JSMainThreadNullState state;
    return IMPL->willValidate();
}

- (NSString *)accessKey
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::accesskeyAttr);
}

- (void)setAccessKey:(NSString *)newAccessKey
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::accesskeyAttr, newAccessKey);
}

- (void)click
{
    WebCore::JSMainThreadNullState state;
    IMPL->click();
}

@end

#undef IMPL
