/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#import "DOMHTMLAnchorElement.h"

#import "DOMNodeInternal.h"
#import "DOMPrivate.h"
#import "DOMTokenListInternal.h"
#import <WebCore/DOMTokenList.h>
#import "ExceptionHandlers.h"
#import <WebCore/HTMLAnchorElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLAnchorElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLAnchorElement

- (NSString *)charset
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::charsetAttr);
}

- (void)setCharset:(NSString *)newCharset
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::charsetAttr, newCharset);
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

- (NSString *)download
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::downloadAttr);
}

- (void)setDownload:(NSString *)newDownload
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::downloadAttr, newDownload);
}

- (NSString *)hreflang
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::hreflangAttr);
}

- (void)setHreflang:(NSString *)newHreflang
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::hreflangAttr, newHreflang);
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

- (NSString *)rev
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::revAttr);
}

- (void)setRev:(NSString *)newRev
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::revAttr, newRev);
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

- (NSString *)text
{
    WebCore::JSMainThreadNullState state;
    return IMPL->text();
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

#undef IMPL
