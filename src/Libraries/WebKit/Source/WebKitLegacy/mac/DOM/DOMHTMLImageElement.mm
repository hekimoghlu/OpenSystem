/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#import "DOMHTMLImageElementInternal.h"

#import "DOMNodeInternal.h"
#import "DOMPrivate.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLImageElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HitTestResult.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLImageElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLImageElement

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

- (NSString *)alt
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::altAttr);
}

- (void)setAlt:(NSString *)newAlt
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::altAttr, newAlt);
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

- (int)height
{
    WebCore::JSMainThreadNullState state;
    return IMPL->height();
}

- (void)setHeight:(int)newHeight
{
    WebCore::JSMainThreadNullState state;
    IMPL->setHeight(newHeight);
}

- (int)hspace
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getIntegralAttribute(WebCore::HTMLNames::hspaceAttr);
}

- (void)setHspace:(int)newHspace
{
    WebCore::JSMainThreadNullState state;
    IMPL->setIntegralAttribute(WebCore::HTMLNames::hspaceAttr, newHspace);
}

- (BOOL)isMap
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::ismapAttr);
}

- (void)setIsMap:(BOOL)newIsMap
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::ismapAttr, newIsMap);
}

- (NSString *)longDesc
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::longdescAttr).string();
}

- (void)setLongDesc:(NSString *)newLongDesc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::longdescAttr, newLongDesc);
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

- (NSString *)srcset
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::srcsetAttr);
}

- (void)setSrcset:(NSString *)newSrcset
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::srcsetAttr, newSrcset);
}

- (NSString *)sizes
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::sizesAttr);
}

- (void)setSizes:(NSString *)newSizes
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::sizesAttr, newSizes);
}

- (NSString *)currentSrc
{
    WebCore::JSMainThreadNullState state;
    return IMPL->currentSrc().string();
}

- (NSString *)useMap
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::usemapAttr);
}

- (void)setUseMap:(NSString *)newUseMap
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::usemapAttr, newUseMap);
}

- (int)vspace
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getIntegralAttribute(WebCore::HTMLNames::vspaceAttr);
}

- (void)setVspace:(int)newVspace
{
    WebCore::JSMainThreadNullState state;
    IMPL->setIntegralAttribute(WebCore::HTMLNames::vspaceAttr, newVspace);
}

- (int)width
{
    WebCore::JSMainThreadNullState state;
    return IMPL->width();
}

- (void)setWidth:(int)newWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setWidth(newWidth);
}

- (BOOL)complete
{
    WebCore::JSMainThreadNullState state;
    return IMPL->complete();
}

- (NSString *)lowsrc
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::lowsrcAttr).string();
}

- (void)setLowsrc:(NSString *)newLowsrc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::lowsrcAttr, newLowsrc);
}

- (int)naturalHeight
{
    WebCore::JSMainThreadNullState state;
    return IMPL->naturalHeight();
}

- (int)naturalWidth
{
    WebCore::JSMainThreadNullState state;
    return IMPL->naturalWidth();
}

- (int)x
{
    WebCore::JSMainThreadNullState state;
    return IMPL->x();
}

- (int)y
{
    WebCore::JSMainThreadNullState state;
    return IMPL->y();
}

- (NSString *)altDisplayString
{
    WebCore::JSMainThreadNullState state;
    return WebCore::displayString(IMPL->alt(), core(self));
}

- (NSURL *)absoluteImageURL
{
    WebCore::JSMainThreadNullState state;
    return [self _getURLAttribute:@"src"];
}

@end

WebCore::HTMLImageElement* core(DOMHTMLImageElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLImageElement*>(wrapper->_internal) : 0;
}

#undef IMPL
