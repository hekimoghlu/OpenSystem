/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#import "DOMHTMLTextAreaElementInternal.h"

#import "DOMHTMLFormElementInternal.h"
#import "DOMNodeInternal.h"
#import "DOMNodeListInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLTextAreaElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/NodeList.h>
#import <WebCore/ThreadCheck.h>

static inline WebCore::HTMLTextAreaElement& unwrap(DOMHTMLTextAreaElement& wrapper)
{
    ASSERT(wrapper._internal);
    return downcast<WebCore::HTMLTextAreaElement>(reinterpret_cast<WebCore::Node&>(*wrapper._internal));
}

WebCore::HTMLTextAreaElement* core(DOMHTMLTextAreaElement *wrapper)
{
    return wrapper ? &unwrap(*wrapper) : nullptr;
}

DOMHTMLTextAreaElement *kit(WebCore::HTMLTextAreaElement* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLTextAreaElement*>(kit(static_cast<WebCore::Node*>(value)));
}

@implementation DOMHTMLTextAreaElement

- (BOOL)autofocus
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).hasAttributeWithoutSynchronization(WebCore::HTMLNames::autofocusAttr);
}

- (void)setAutofocus:(BOOL)newAutofocus
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setBooleanAttribute(WebCore::HTMLNames::autofocusAttr, newAutofocus);
}

- (NSString *)dirName
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).getAttribute(WebCore::HTMLNames::dirnameAttr);
}

- (void)setDirName:(NSString *)newDirName
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setAttributeWithoutSynchronization(WebCore::HTMLNames::dirnameAttr, newDirName);
}

- (BOOL)disabled
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).hasAttributeWithoutSynchronization(WebCore::HTMLNames::disabledAttr);
}

- (void)setDisabled:(BOOL)newDisabled
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setBooleanAttribute(WebCore::HTMLNames::disabledAttr, newDisabled);
}

- (DOMHTMLFormElement *)form
{
    WebCore::JSMainThreadNullState state;
    return kit(unwrap(*self).form());
}

- (int)maxLength
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).maxLength();
}

- (void)setMaxLength:(int)newMaxLength
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(unwrap(*self).setMaxLength(newMaxLength));
}

- (NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).getNameAttribute();
}

- (void)setName:(NSString *)newName
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setAttributeWithoutSynchronization(WebCore::HTMLNames::nameAttr, newName);
}

- (NSString *)placeholder
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).getAttribute(WebCore::HTMLNames::placeholderAttr);
}

- (void)setPlaceholder:(NSString *)newPlaceholder
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setAttributeWithoutSynchronization(WebCore::HTMLNames::placeholderAttr, newPlaceholder);
}

- (BOOL)readOnly
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).hasAttributeWithoutSynchronization(WebCore::HTMLNames::readonlyAttr);
}

- (void)setReadOnly:(BOOL)newReadOnly
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setBooleanAttribute(WebCore::HTMLNames::readonlyAttr, newReadOnly);
}

- (BOOL)required
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).hasAttributeWithoutSynchronization(WebCore::HTMLNames::requiredAttr);
}

- (void)setRequired:(BOOL)newRequired
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setBooleanAttribute(WebCore::HTMLNames::requiredAttr, newRequired);
}

- (int)rows
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).rows();
}

- (void)setRows:(int)newRows
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setRows(newRows);
}

- (int)cols
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).cols();
}

- (void)setCols:(int)newCols
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setCols(newCols);
}

- (NSString *)wrap
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).getAttribute(WebCore::HTMLNames::wrapAttr);
}

- (void)setWrap:(NSString *)newWrap
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setAttributeWithoutSynchronization(WebCore::HTMLNames::wrapAttr, newWrap);
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).type();
}

- (NSString *)defaultValue
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).defaultValue();
}

- (void)setDefaultValue:(NSString *)newDefaultValue
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setDefaultValue(newDefaultValue);
}

- (NSString *)value
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).value();
}

- (void)setValue:(NSString *)newValue
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setValue(newValue);
}

- (unsigned)textLength
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).textLength();
}

- (BOOL)willValidate
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).willValidate();
}

- (DOMNodeList *)labels
{
    WebCore::JSMainThreadNullState state;
    return kit(unwrap(*self).labels().get());
}

- (int)selectionStart
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).selectionStart();
}

- (void)setSelectionStart:(int)newSelectionStart
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setSelectionStart(newSelectionStart);
}

- (int)selectionEnd
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).selectionEnd();
}

- (void)setSelectionEnd:(int)newSelectionEnd
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setSelectionEnd(newSelectionEnd);
}

- (NSString *)selectionDirection
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).selectionDirection();
}

- (void)setSelectionDirection:(NSString *)newSelectionDirection
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setSelectionDirection(newSelectionDirection);
}

- (NSString *)accessKey
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).getAttribute(WebCore::HTMLNames::accesskeyAttr);
}

- (void)setAccessKey:(NSString *)newAccessKey
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setAttributeWithoutSynchronization(WebCore::HTMLNames::accesskeyAttr, newAccessKey);
}

- (NSString *)autocomplete
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).autocomplete();
}

- (void)setAutocomplete:(NSString *)newAutocomplete
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setAutocomplete(newAutocomplete);
}

- (void)select
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).select();
}

- (void)setRangeText:(NSString *)replacement
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(unwrap(*self).setRangeText(String { replacement }));
}

- (void)setRangeText:(NSString *)replacement start:(unsigned)start end:(unsigned)end selectionMode:(NSString *)selectionMode
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(unwrap(*self).setRangeText(String { replacement }, start, end, selectionMode));
}

- (void)setSelectionRange:(int)start end:(int)end
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setSelectionRange(start, end);
}

- (BOOL)canShowPlaceholder
{
    WebCore::JSMainThreadNullState state;
    return unwrap(*self).canShowPlaceholder();
}

- (void)setCanShowPlaceholder:(BOOL)canShowPlaceholder
{
    WebCore::JSMainThreadNullState state;
    unwrap(*self).setCanShowPlaceholder(canShowPlaceholder);
}

@end
