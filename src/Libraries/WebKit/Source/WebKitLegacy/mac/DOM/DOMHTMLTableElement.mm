/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#import "DOMHTMLTableElement.h"

#import "DOMHTMLCollectionInternal.h"
#import "DOMHTMLElementInternal.h"
#import "DOMHTMLTableCaptionElementInternal.h"
#import "DOMHTMLTableSectionElementInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/HTMLCollection.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLTableCaptionElement.h>
#import <WebCore/HTMLTableElement.h>
#import <WebCore/HTMLTableSectionElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLTableElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLTableElement

- (DOMHTMLTableCaptionElement *)caption
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->caption()));
}

- (void)setCaption:(DOMHTMLTableCaptionElement *)newCaption
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setCaption(core(newCaption)));
}

- (DOMHTMLTableSectionElement *)tHead
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->tHead()));
}

- (void)setTHead:(DOMHTMLTableSectionElement *)newTHead
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setTHead(core(newTHead)));
}

- (DOMHTMLTableSectionElement *)tFoot
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->tFoot()));
}

- (void)setTFoot:(DOMHTMLTableSectionElement *)newTFoot
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setTFoot(core(newTFoot)));
}

- (DOMHTMLCollection *)rows
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->rows()));
}

- (DOMHTMLCollection *)tBodies
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->tBodies()));
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

- (NSString *)border
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::borderAttr);
}

- (void)setBorder:(NSString *)newBorder
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::borderAttr, newBorder);
}

- (NSString *)cellPadding
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::cellpaddingAttr);
}

- (void)setCellPadding:(NSString *)newCellPadding
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::cellpaddingAttr, newCellPadding);
}

- (NSString *)cellSpacing
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::cellspacingAttr);
}

- (void)setCellSpacing:(NSString *)newCellSpacing
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::cellspacingAttr, newCellSpacing);
}

- (NSString *)frameBorders
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::frameAttr);
}

- (void)setFrameBorders:(NSString *)newFrameBorders
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::frameAttr, newFrameBorders);
}

- (NSString *)rules
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::rulesAttr);
}

- (void)setRules:(NSString *)newRules
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::rulesAttr, newRules);
}

- (NSString *)summary
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::summaryAttr);
}

- (void)setSummary:(NSString *)newSummary
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::summaryAttr, newSummary);
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

- (DOMHTMLElement *)createTHead
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->createTHead()));
}

- (void)deleteTHead
{
    WebCore::JSMainThreadNullState state;
    IMPL->deleteTHead();
}

- (DOMHTMLElement *)createTFoot
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->createTFoot()));
}

- (void)deleteTFoot
{
    WebCore::JSMainThreadNullState state;
    IMPL->deleteTFoot();
}

- (DOMHTMLElement *)createTBody
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->createTBody()));
}

- (DOMHTMLElement *)createCaption
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->createCaption()));
}

- (void)deleteCaption
{
    WebCore::JSMainThreadNullState state;
    IMPL->deleteCaption();
}

- (DOMHTMLElement *)insertRow:(int)index
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->insertRow(index)).ptr());
}

- (void)deleteRow:(int)index
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->deleteRow(index));
}

@end

#undef IMPL
