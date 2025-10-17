/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#import "DOMCSSPrimitiveValueInternal.h"

#import <WebCore/DeprecatedCSSOMCounter.h>
#import <WebCore/DeprecatedCSSOMPrimitiveValue.h>
#import <WebCore/DeprecatedCSSOMRGBColor.h>
#import <WebCore/DeprecatedCSSOMRect.h>
#import "DOMCSSValueInternal.h"
#import "DOMCounterInternal.h"
#import "DOMNodeInternal.h"
#import "DOMRGBColorInternal.h"
#import "DOMRectInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::DeprecatedCSSOMPrimitiveValue*>(reinterpret_cast<WebCore::DeprecatedCSSOMValue*>(_internal))

@implementation DOMCSSPrimitiveValue

- (unsigned short)primitiveType
{
    WebCore::JSMainThreadNullState state;
    return IMPL->primitiveType();
}

- (void)setFloatValue:(unsigned short)unitType floatValue:(float)floatValue
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setFloatValue(unitType, floatValue));
}

- (float)getFloatValue:(unsigned short)unitType
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->getFloatValue(unitType));
}

- (void)setStringValue:(unsigned short)stringType stringValue:(NSString *)stringValue
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setStringValue(stringType, stringValue));
}

- (NSString *)getStringValue
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->getStringValue());
}

- (DOMCounter *)getCounterValue
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->getCounterValue()).ptr());
}

- (DOMRect *)getRectValue
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->getRectValue()).ptr());
}

- (DOMRGBColor *)getRGBColorValue
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->getRGBColorValue()).ptr());
}

@end

@implementation DOMCSSPrimitiveValue (DOMCSSPrimitiveValueDeprecated)

- (void)setFloatValue:(unsigned short)unitType :(float)floatValue
{
    [self setFloatValue:unitType floatValue:floatValue];
}

- (void)setStringValue:(unsigned short)stringType :(NSString *)stringValue
{
    [self setStringValue:stringType stringValue:stringValue];
}

@end

DOMCSSPrimitiveValue *kit(WebCore::DeprecatedCSSOMPrimitiveValue* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMCSSPrimitiveValue*>(kit(static_cast<WebCore::DeprecatedCSSOMValue*>(value)));
}

#undef IMPL
