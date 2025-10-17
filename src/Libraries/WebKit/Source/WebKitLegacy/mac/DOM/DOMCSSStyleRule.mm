/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#import "DOMCSSStyleRule.h"

#import <WebCore/CSSStyleDeclaration.h>
#import <WebCore/CSSStyleRule.h>
#import "DOMCSSRuleInternal.h"
#import "DOMCSSStyleDeclarationInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/StyleProperties.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::CSSStyleRule*>(reinterpret_cast<WebCore::CSSRule*>(_internal))

@implementation DOMCSSStyleRule

- (NSString *)selectorText
{
    WebCore::JSMainThreadNullState state;
    return IMPL->selectorText();
}

- (void)setSelectorText:(NSString *)newSelectorText
{
    WebCore::JSMainThreadNullState state;
    IMPL->setSelectorText(newSelectorText);
}

- (DOMCSSStyleDeclaration *)style
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->style()));
}

@end

#undef IMPL
