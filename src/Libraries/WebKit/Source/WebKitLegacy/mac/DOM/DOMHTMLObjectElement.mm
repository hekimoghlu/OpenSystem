/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#import "DOMHTMLObjectElement.h"

#import "DOMDocumentInternal.h"
#import "DOMHTMLFormElementInternal.h"
#import "DOMNodeInternal.h"
#import "DOMPrivate.h"
#import <WebCore/Document.h>
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLFormElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLObjectElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/RenderElement.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLObjectElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLObjectElement

- (DOMHTMLFormElement *)form
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->form()));
}

- (NSString *)code
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::codeAttr);
}

- (void)setCode:(NSString *)newCode
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::codeAttr, newCode);
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

- (NSString *)archive
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::archiveAttr);
}

- (void)setArchive:(NSString *)newArchive
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::archiveAttr, newArchive);
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

- (NSString *)codeBase
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::codebaseAttr);
}

- (void)setCodeBase:(NSString *)newCodeBase
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::codebaseAttr, newCodeBase);
}

- (NSString *)codeType
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::codetypeAttr);
}

- (void)setCodeType:(NSString *)newCodeType
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::codetypeAttr, newCodeType);
}

- (NSString *)data
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::dataAttr).string();
}

- (void)setData:(NSString *)newData
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::dataAttr, newData);
}

- (BOOL)declare
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::declareAttr);
}

- (void)setDeclare:(BOOL)newDeclare
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::declareAttr, newDeclare);
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

- (NSString *)standby
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::standbyAttr);
}

- (void)setStandby:(NSString *)newStandby
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::standbyAttr, newStandby);
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

- (DOMDocument *)contentDocument
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->contentDocument()));
}

- (NSURL *)absoluteImageURL
{
    WebCore::JSMainThreadNullState state;
    if (!IMPL->renderer() || !IMPL->renderer()->isImage())
        return nil;
    return [self _getURLAttribute:@"data"];
}

@end

#undef IMPL
