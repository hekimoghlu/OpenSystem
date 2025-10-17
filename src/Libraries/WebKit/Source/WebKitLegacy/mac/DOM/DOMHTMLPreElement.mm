/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#import "DOMHTMLPreElement.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLPreElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#define IMPL static_cast<WebCore::HTMLPreElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLPreElement

- (int)width
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getIntegralAttribute(WebCore::HTMLNames::widthAttr);
}

- (void)setWidth:(int)newWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setIntegralAttribute(WebCore::HTMLNames::widthAttr, newWidth);
}

- (BOOL)wrap
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::wrapAttr);
}

- (void)setWrap:(BOOL)newWrap
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::wrapAttr, newWrap);
}

@end

#undef IMPL
