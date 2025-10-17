/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#import "DOMHTMLSelectElementInternal.h"

#import "DOMHTMLCollectionInternal.h"
#import "DOMHTMLElementInternal.h"
#import "DOMHTMLFormElementInternal.h"
#import "DOMHTMLOptionsCollectionInternal.h"
#import "DOMNodeInternal.h"
#import "DOMNodeListInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLCollection.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/HTMLFormElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLOptGroupElement.h>
#import <WebCore/HTMLOptionsCollection.h>
#import <WebCore/HTMLSelectElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/Node.h>
#import <WebCore/NodeList.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLSelectElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLSelectElement

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

- (BOOL)multiple
{
    WebCore::JSMainThreadNullState state;
    return IMPL->multiple();
}

- (void)setMultiple:(BOOL)newMultiple
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::multipleAttr, newMultiple);
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

- (int)size
{
    WebCore::JSMainThreadNullState state;
    return IMPL->size();
}

- (void)setSize:(int)newSize
{
    WebCore::JSMainThreadNullState state;
    IMPL->setSize(newSize);
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->type();
}

- (DOMHTMLOptionsCollection *)options
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->options()));
}

- (int)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (int)selectedIndex
{
    WebCore::JSMainThreadNullState state;
    return IMPL->selectedIndex();
}

- (void)setSelectedIndex:(int)newSelectedIndex
{
    WebCore::JSMainThreadNullState state;
    IMPL->setSelectedIndex(newSelectedIndex);
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

- (BOOL)willValidate
{
    WebCore::JSMainThreadNullState state;
    return IMPL->willValidate();
}

- (DOMNode *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->item(index)));
}

- (DOMNode *)namedItem:(NSString *)inName
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->namedItem(inName)));
}

- (void)add:(DOMHTMLElement *)element before:(DOMHTMLElement *)before
{
    WebCore::JSMainThreadNullState state;
    if (!element)
        raiseTypeErrorException();

    auto& coreElement = *core(element);
    std::variant<RefPtr<WebCore::HTMLOptionElement>, RefPtr<WebCore::HTMLOptGroupElement>> variantElement;
    if (is<WebCore::HTMLOptionElement>(coreElement))
        variantElement = &downcast<WebCore::HTMLOptionElement>(coreElement);
    else if (is<WebCore::HTMLOptGroupElement>(coreElement))
        variantElement = &downcast<WebCore::HTMLOptGroupElement>(coreElement);
    else
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->add(WTFMove(variantElement), WebCore::HTMLSelectElement::HTMLElementOrInt(core(before))));
}

- (void)remove:(int)index
{
    WebCore::JSMainThreadNullState state;
    IMPL->remove(index);
}

@end

@implementation DOMHTMLSelectElement (DOMHTMLSelectElementDeprecated)

- (void)add:(DOMHTMLElement *)element :(DOMHTMLElement *)before
{
    [self add:element before:before];
}

@end

WebCore::HTMLSelectElement* core(DOMHTMLSelectElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLSelectElement*>(wrapper->_internal) : 0;
}

#undef IMPL
