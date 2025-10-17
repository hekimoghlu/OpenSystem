/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#import "DOMHTMLTableCellElementInternal.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLTableCellElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLTableCellElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLTableCellElement

- (int)cellIndex
{
    WebCore::JSMainThreadNullState state;
    return IMPL->cellIndex();
}

- (NSString *)align
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::alignAttr);
}

- (void)setAlign:(NSString *)newAlign
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::alignAttr, newAlign);
}

- (NSString *)axis
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::axisAttr);
}

- (void)setAxis:(NSString *)newAxis
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::axisAttr, newAxis);
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

- (NSString *)ch
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::charAttr);
}

- (void)setCh:(NSString *)newCh
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::charAttr, newCh);
}

- (NSString *)chOff
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::charoffAttr);
}

- (void)setChOff:(NSString *)newChOff
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::charoffAttr, newChOff);
}

- (int)colSpan
{
    WebCore::JSMainThreadNullState state;
    return IMPL->colSpan();
}

- (void)setColSpan:(int)newColSpan
{
    WebCore::JSMainThreadNullState state;
    IMPL->setColSpan(newColSpan);
}

- (int)rowSpan
{
    WebCore::JSMainThreadNullState state;
    return IMPL->rowSpanForBindings();
}

- (void)setRowSpan:(int)newRowSpan
{
    WebCore::JSMainThreadNullState state;
    IMPL->setRowSpanForBindings(newRowSpan);
}

- (NSString *)headers
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::headersAttr);
}

- (void)setHeaders:(NSString *)newHeaders
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::headersAttr, newHeaders);
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

- (BOOL)noWrap
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::nowrapAttr);
}

- (void)setNoWrap:(BOOL)newNoWrap
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::nowrapAttr, newNoWrap);
}

- (NSString *)vAlign
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::valignAttr);
}

- (void)setVAlign:(NSString *)newVAlign
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::valignAttr, newVAlign);
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

- (NSString *)abbr
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::abbrAttr);
}

- (void)setAbbr:(NSString *)newAbbr
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::abbrAttr, newAbbr);
}

- (NSString *)scope
{
    WebCore::JSMainThreadNullState state;
    return IMPL->scope();
}

- (void)setScope:(NSString *)newScope
{
    WebCore::JSMainThreadNullState state;
    IMPL->setScope(newScope);
}

@end

WebCore::HTMLTableCellElement* core(DOMHTMLTableCellElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLTableCellElement*>(wrapper->_internal) : 0;
}

DOMHTMLTableCellElement *kit(WebCore::HTMLTableCellElement* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLTableCellElement*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
