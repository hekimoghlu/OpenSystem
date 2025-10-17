/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#import "DOMCSSMediaRule.h"

#import <WebCore/CSSMediaRule.h>
#import <WebCore/CSSRuleList.h>
#import "DOMCSSRuleInternal.h"
#import "DOMCSSRuleListInternal.h"
#import "DOMMediaListInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/MediaList.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::CSSMediaRule*>(reinterpret_cast<WebCore::CSSRule*>(_internal))

@implementation DOMCSSMediaRule

- (DOMMediaList *)media
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->media()));
}

- (DOMCSSRuleList *)cssRules
{
    WebCore::JSMainThreadNullState state;
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
    return raiseOnDOMError(IMPL->deleteRule(index));
}

@end

@implementation DOMCSSMediaRule (DOMCSSMediaRuleDeprecated)

- (unsigned)insertRule:(NSString *)rule :(unsigned)index
{
    return [self insertRule:rule index:index];
}

@end

#undef IMPL
