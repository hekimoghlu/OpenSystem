/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
#import "DOMAttrInternal.h"

#import <WebCore/Attr.h>
#import <WebCore/CSSStyleDeclaration.h>
#import "DOMCSSStyleDeclarationInternal.h"
#import "DOMElementInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/Element.h>
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/StyleProperties.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::Attr*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMAttr

- (NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return IMPL->name();
}

- (BOOL)specified
{
    WebCore::JSMainThreadNullState state;
    return IMPL->specified();
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

- (DOMElement *)ownerElement
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->ownerElement()));
}

- (BOOL)isId
{
    WebCore::JSMainThreadNullState state;
    return IMPL->qualifiedName().matches(WebCore::HTMLNames::idAttr);
}

- (DOMCSSStyleDeclaration *)style
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->style()));
}

@end

WebCore::Attr* core(DOMAttr *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::Attr*>(wrapper->_internal) : 0;
}

DOMAttr *kit(WebCore::Attr* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMAttr*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
