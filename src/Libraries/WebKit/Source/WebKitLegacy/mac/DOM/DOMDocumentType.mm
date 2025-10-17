/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#import "DOMDocumentTypeInternal.h"

#import "DOMNamedNodeMapInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/DocumentType.h>
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/NamedNodeMap.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::DocumentType*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMDocumentType

- (NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return IMPL->name();
}

- (DOMNamedNodeMap *)entities
{
    return nil;
}

- (DOMNamedNodeMap *)notations
{
    return nil;
}

- (NSString *)publicId
{
    WebCore::JSMainThreadNullState state;
    return IMPL->publicId();
}

- (NSString *)systemId
{
    WebCore::JSMainThreadNullState state;
    return IMPL->systemId();
}

- (NSString *)internalSubset
{
    return @"";
}

- (void)remove
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->remove());
}

@end

WebCore::DocumentType* core(DOMDocumentType *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::DocumentType*>(wrapper->_internal) : 0;
}

DOMDocumentType *kit(WebCore::DocumentType* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMDocumentType*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
