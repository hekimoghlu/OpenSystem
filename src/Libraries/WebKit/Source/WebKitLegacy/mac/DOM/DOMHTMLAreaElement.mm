/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#import "DOMHTMLAreaElementInternal.h"

#import "DOMNodeInternal.h"
#import "DOMPrivate.h"
#import "DOMTokenListInternal.h"
#import <WebCore/DOMTokenList.h>
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLAreaElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLAreaElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLAreaElement

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

- (NSString *)coords
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::coordsAttr);
}

- (void)setCoords:(NSString *)newCoords
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::coordsAttr, newCoords);
}

- (BOOL)noHref
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::nohrefAttr);
}

- (void)setNoHref:(BOOL)newNoHref
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::nohrefAttr, newNoHref);
}

- (NSString *)ping
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::pingAttr);
}

- (void)setPing:(NSString *)newPing
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::pingAttr, newPing);
}

- (NSString *)rel
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::relAttr);
}

- (void)setRel:(NSString *)newRel
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::relAttr, newRel);
}

- (NSString *)shape
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::shapeAttr);
}

- (void)setShape:(NSString *)newShape
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::shapeAttr, newShape);
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

- (NSURL *)absoluteLinkURL
{
    WebCore::JSMainThreadNullState state;
    return [self _getURLAttribute:@"href"];
}

- (DOMTokenList *)relList
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->relList()));
}

- (NSString *)href
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::hrefAttr).string();
}

- (void)setHref:(NSString *)newHref
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::hrefAttr, newHref);
}

- (NSString *)origin
{
    WebCore::JSMainThreadNullState state;
    return IMPL->origin();
}

- (NSString *)protocol
{
    WebCore::JSMainThreadNullState state;
    return IMPL->protocol();
}

- (NSString *)host
{
    WebCore::JSMainThreadNullState state;
    return IMPL->host();
}

- (NSString *)hostname
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hostname();
}

- (NSString *)port
{
    WebCore::JSMainThreadNullState state;
    return IMPL->port();
}

- (NSString *)pathname
{
    WebCore::JSMainThreadNullState state;
    return IMPL->pathname();
}

- (NSString *)search
{
    WebCore::JSMainThreadNullState state;
    return IMPL->search();
}

- (NSString *)hashName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hash();
}

@end

WebCore::HTMLAreaElement* core(DOMHTMLAreaElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLAreaElement*>(wrapper->_internal) : 0;
}

#undef IMPL
