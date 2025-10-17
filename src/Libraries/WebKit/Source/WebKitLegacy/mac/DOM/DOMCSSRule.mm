/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#import "DOMCSSRuleInternal.h"

#import <WebCore/CSSImportRule.h>
#import <WebCore/CSSRule.h>
#import <WebCore/CSSStyleSheet.h>
#import "DOMInternal.h"
#import "DOMCSSStyleSheetInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::CSSRule*>(_internal)

@implementation DOMCSSRule

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMCSSRule class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (unsigned short)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->typeForCSSOM();
}

- (NSString *)cssText
{
    WebCore::JSMainThreadNullState state;
    return IMPL->cssText();
}

- (void)setCssText:(NSString *)newCssText
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setCssText(newCssText));
}

- (DOMCSSStyleSheet *)parentStyleSheet
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->parentStyleSheet()));
}

- (DOMCSSRule *)parentRule
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->parentRule()));
}

@end

DOMCSSRule *kit(WebCore::CSSRule* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMCSSRule *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    RetainPtr<DOMCSSRule> wrapper = adoptNS([[kitClass(value) alloc] _init]);
    if (!wrapper)
        return nil;
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
