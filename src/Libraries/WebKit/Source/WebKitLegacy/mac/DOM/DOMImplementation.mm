/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#import "DOMImplementationInternal.h"

#import "DOMCSSStyleSheetInternal.h"
#import "DOMDocumentInternal.h"
#import "DOMDocumentTypeInternal.h"
#import "DOMHTMLDocumentInternal.h"
#import "DOMInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/CSSStyleSheet.h>
#import <WebCore/DOMImplementation.h>
#import <WebCore/DocumentType.h>
#import <WebCore/HTMLDocument.h>
#import <WebCore/JSExecState.h>
#import <WebCore/SVGTests.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>

@implementation DOMImplementation

static inline WebCore::DOMImplementation& unwrap(DOMImplementation& wrapper)
{
    return *reinterpret_cast<WebCore::DOMImplementation*>(wrapper._internal);
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMImplementation class], self))
        return;
    if (_internal)
        unwrap(*self).deref();
    [super dealloc];
}

- (BOOL)hasFeature:(NSString *)feature version:(NSString *)version
{
    return YES;
}

- (DOMDocumentType *)createDocumentType:(NSString *)qualifiedName publicId:(NSString *)publicId systemId:(NSString *)systemId
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(unwrap(*self).createDocumentType(qualifiedName, publicId, systemId)).ptr());
}

- (DOMDocument *)createDocument:(NSString *)namespaceURI qualifiedName:(NSString *)qualifiedName doctype:(DOMDocumentType *)doctype
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(unwrap(*self).createDocument(namespaceURI, qualifiedName, core(doctype))).ptr());
}

- (DOMCSSStyleSheet *)createCSSStyleSheet:(NSString *)title media:(NSString *)media
{
    WebCore::JSMainThreadNullState state;
    return kit(unwrap(*self).createCSSStyleSheet(title, media).ptr());
}

- (DOMHTMLDocument *)createHTMLDocument:(NSString *)title
{
    WebCore::JSMainThreadNullState state;
    return kit(unwrap(*self).createHTMLDocument(title).ptr());
}

@end

@implementation DOMImplementation (DOMImplementationDeprecated)

- (BOOL)hasFeature:(NSString *)feature :(NSString *)version
{
    return [self hasFeature:feature version:version];
}

- (DOMDocumentType *)createDocumentType:(NSString *)qualifiedName :(NSString *)publicId :(NSString *)systemId
{
    return [self createDocumentType:qualifiedName publicId:publicId systemId:systemId];
}

- (DOMDocument *)createDocument:(NSString *)namespaceURI :(NSString *)qualifiedName :(DOMDocumentType *)doctype
{
    return [self createDocument:namespaceURI qualifiedName:qualifiedName doctype:doctype];
}

- (DOMCSSStyleSheet *)createCSSStyleSheet:(NSString *)title :(NSString *)media
{
    return [self createCSSStyleSheet:title media:media];
}

@end

DOMImplementation *kit(WebCore::DOMImplementation* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMImplementation *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMImplementation alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}
