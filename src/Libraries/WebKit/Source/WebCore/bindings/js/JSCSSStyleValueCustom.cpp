/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
#include "config.h"
#include "JSCSSStyleValue.h"

#include "JSCSSKeywordValue.h"
#include "JSCSSMathClamp.h"
#include "JSCSSMathInvert.h"
#include "JSCSSMathMax.h"
#include "JSCSSMathMin.h"
#include "JSCSSMathNegate.h"
#include "JSCSSMathProduct.h"
#include "JSCSSMathSum.h"
#include "JSCSSMathValue.h"
#include "JSCSSNumericValue.h"
#include "JSCSSStyleImageValue.h"
#include "JSCSSTransformValue.h"
#include "JSCSSUnitValue.h"
#include "JSCSSUnparsedValue.h"
#include "JSDOMWrapperCache.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<CSSStyleValue>&& value)
{
    switch (value->getType()) {
    case CSSStyleValueType::CSSStyleImageValue:
        return createWrapper<CSSStyleImageValue>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathClamp:
        return createWrapper<CSSMathClamp>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathInvert:
        return createWrapper<CSSMathInvert>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathMin:
        return createWrapper<CSSMathMin>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathMax:
        return createWrapper<CSSMathMax>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathNegate:
        return createWrapper<CSSMathNegate>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathProduct:
        return createWrapper<CSSMathProduct>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSMathSum:
        return createWrapper<CSSMathSum>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSUnitValue:
        return createWrapper<CSSUnitValue>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSUnparsedValue:
        return createWrapper<CSSUnparsedValue>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSKeywordValue:
        return createWrapper<CSSKeywordValue>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSTransformValue:
        return createWrapper<CSSTransformValue>(globalObject, WTFMove(value));
    case CSSStyleValueType::CSSStyleValue:
        return createWrapper<CSSStyleValue>(globalObject, WTFMove(value));
    }

    ASSERT_NOT_REACHED();
    return createWrapper<CSSStyleValue>(globalObject, WTFMove(value));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, CSSStyleValue& object)
{
    return wrap(lexicalGlobalObject, globalObject, object);
}

} // namespace WebCore
