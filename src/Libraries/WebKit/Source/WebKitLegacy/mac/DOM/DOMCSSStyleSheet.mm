/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
#import "DOMCSSStyleSheetInternal.h"

#import <WebCore/CSSImportRule.h>
#import <WebCore/CSSRule.h>
#import <WebCore/CSSRuleList.h>
#import <WebCore/CSSStyleSheet.h>
#import "DOMCSSRuleInternal.h"
#import "DOMCSSRuleListInternal.h"
#import "DOMNodeInternal.h"
#import "DOMStyleSheetInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::CSSStyleSheet*>(reinterpret_cast<WebCore::StyleSheet*>(_internal))

@implementation DOMCSSStyleSheet

- (DOMCSSRule *)ownerRule
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->ownerRule()));
}

- (DOMCSSRuleList *)cssRules
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->cssRules()));
}

- (DOMCSSRuleList *)rules
{
    WebCore::JSMainThreadNullState state;
    // Calling IMPL->cssRules (not IMPL->rules) is intentional, as `rules` should just be an alias for `cssRules`.
    // See https://bugs.webkit.org/show_bug.cgi?id=197725 for more information.
    return kit(WTF::getPtr(IMPL->cssRules()));
}

- (unsigned)insertRule:(NSString *)rule index:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->insertRule(rule, index));
}

- (void)deleteRule:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->deleteRule(index));
}

- (int)addRule:(NSString *)selector style:(NSString *)style index:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->addRule(selector, style, index));
}

- (void)removeRule:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->removeRule(index));
}

@end

@implementation DOMCSSStyleSheet (DOMCSSStyleSheetDeprecated)

- (unsigned)insertRule:(NSString *)rule :(unsigned)index
{
    return [self insertRule:rule index:index];
}

@end

DOMCSSStyleSheet *kit(WebCore::CSSStyleSheet* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMCSSStyleSheet*>(kit(static_cast<WebCore::StyleSheet*>(value)));
}

#undef IMPL
