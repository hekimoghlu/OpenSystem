/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#import "DOMHTMLOptionElementInternal.h"

#import "DOMHTMLFormElementInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLFormElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLOptionElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLOptionElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLOptionElement

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

- (NSString *)label
{
    WebCore::JSMainThreadNullState state;
    return IMPL->label();
}

- (void)setLabel:(NSString *)newLabel
{
    WebCore::JSMainThreadNullState state;
    IMPL->setLabel(newLabel);
}

- (BOOL)defaultSelected
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::selectedAttr);
}

- (void)setDefaultSelected:(BOOL)newDefaultSelected
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::selectedAttr, newDefaultSelected);
}

- (BOOL)selected
{
    WebCore::JSMainThreadNullState state;
    return IMPL->selected();
}

- (void)setSelected:(BOOL)newSelected
{
    WebCore::JSMainThreadNullState state;
    IMPL->setSelected(newSelected);
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

- (NSString *)text
{
    WebCore::JSMainThreadNullState state;
    return IMPL->text();
}

- (int)index
{
    WebCore::JSMainThreadNullState state;
    return IMPL->index();
}

@end

WebCore::HTMLOptionElement* core(DOMHTMLOptionElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLOptionElement*>(wrapper->_internal) : 0;
}

DOMHTMLOptionElement *kit(WebCore::HTMLOptionElement* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLOptionElement*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
