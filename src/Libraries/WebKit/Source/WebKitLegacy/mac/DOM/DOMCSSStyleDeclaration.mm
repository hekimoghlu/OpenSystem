/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
#import "DOMCSSStyleDeclarationInternal.h"

#import <WebCore/CSSImportRule.h>
#import <WebCore/CSSRule.h>
#import <WebCore/CSSStyleDeclaration.h>
#import <WebCore/DeprecatedCSSOMValue.h>
#import "DOMCSSRuleInternal.h"
#import "DOMCSSValueInternal.h"
#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::CSSStyleDeclaration*>(_internal)

@implementation DOMCSSStyleDeclaration

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMCSSStyleDeclaration class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
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

- (unsigned)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (DOMCSSRule *)parentRule
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->parentRule()));
}

- (NSString *)getPropertyValue:(NSString *)propertyName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getPropertyValue(propertyName);
}

- (DOMCSSValue *)getPropertyCSSValue:(NSString *)propertyName
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->getPropertyCSSValue(propertyName)));
}

- (NSString *)removeProperty:(NSString *)propertyName
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->removeProperty(propertyName));
}

- (NSString *)getPropertyPriority:(NSString *)propertyName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getPropertyPriority(propertyName);
}

- (void)setProperty:(NSString *)propertyName value:(NSString *)value priority:(NSString *)priority
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->setProperty(propertyName, value, priority));
}

- (NSString *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return IMPL->item(index);
}

- (NSString *)getPropertyShorthand:(NSString *)propertyName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getPropertyShorthand(propertyName);
}

- (BOOL)isPropertyImplicit:(NSString *)propertyName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->isPropertyImplicit(propertyName);
}

@end

@implementation DOMCSSStyleDeclaration (DOMCSSStyleDeclarationDeprecated)

- (void)setProperty:(NSString *)propertyName :(NSString *)value :(NSString *)priority
{
    [self setProperty:propertyName value:value priority:priority];
}

@end

WebCore::CSSStyleDeclaration* core(DOMCSSStyleDeclaration *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::CSSStyleDeclaration*>(wrapper->_internal) : 0;
}

DOMCSSStyleDeclaration *kit(WebCore::CSSStyleDeclaration* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMCSSStyleDeclaration *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMCSSStyleDeclaration alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
