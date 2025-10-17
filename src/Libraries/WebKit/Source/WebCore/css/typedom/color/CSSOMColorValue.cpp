/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "CSSOMColorValue.h"

#include "CSSKeywordValue.h"
#include "CSSUnitValue.h"
#include "CSSUnits.h"
#include "ExceptionOr.h"

namespace WebCore {

RefPtr<CSSKeywordValue> CSSOMColorValue::colorSpace()
{
    // FIXME: implement this.
    return nullptr;
}

RefPtr<CSSOMColorValue> CSSOMColorValue::to(CSSKeywordish)
{
    // FIXME: implement this.
    return nullptr;
}

std::variant<RefPtr<CSSOMColorValue>, RefPtr<CSSStyleValue>> CSSOMColorValue::parse(const String&)
{
    // FIXME: implement this.
    return RefPtr<CSSOMColorValue> { nullptr };
}

// https://drafts.css-houdini.org/css-typed-om-1/#rectify-a-csscolorpercent
ExceptionOr<RectifiedCSSColorPercent> CSSOMColorValue::rectifyCSSColorPercent(CSSColorPercent&& colorPercent)
{
    return switchOn(WTFMove(colorPercent), [](double value) -> ExceptionOr<RectifiedCSSColorPercent> {
        return { RefPtr<CSSNumericValue> { CSSUnitValue::create(value * 100, CSSUnitType::CSS_PERCENTAGE) } };
    }, [](RefPtr<CSSNumericValue>&& numericValue) -> ExceptionOr<RectifiedCSSColorPercent> {
        if (numericValue->type().matches<CSSNumericBaseType::Percent>())
            return { WTFMove(numericValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorPercent"_s };
    }, [](String&& string) -> ExceptionOr<RectifiedCSSColorPercent> {
        return { RefPtr<CSSKeywordValue> { CSSKeywordValue::rectifyKeywordish(WTFMove(string)) } };
    }, [](RefPtr<CSSKeywordValue>&& keywordValue) -> ExceptionOr<RectifiedCSSColorPercent> {
        if (equalIgnoringASCIICase(keywordValue->value(), "none"_s))
            return { WTFMove(keywordValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorPercent"_s };
    });
}

// https://drafts.css-houdini.org/css-typed-om/#rectify-a-csscolorangle
ExceptionOr<RectifiedCSSColorAngle> CSSOMColorValue::rectifyCSSColorAngle(CSSColorAngle&& colorAngle)
{
    return switchOn(WTFMove(colorAngle), [](double value) -> ExceptionOr<RectifiedCSSColorAngle> {
        return { RefPtr<CSSNumericValue> { CSSUnitValue::create(value, CSSUnitType::CSS_DEG) } };
    }, [](RefPtr<CSSNumericValue>&& numericValue) -> ExceptionOr<RectifiedCSSColorAngle> {
        if (numericValue->type().matches<CSSNumericBaseType::Angle>())
            return { WTFMove(numericValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorAngle"_s };
    }, [](String&& string) -> ExceptionOr<RectifiedCSSColorAngle> {
        return { RefPtr<CSSKeywordValue> { CSSKeywordValue::rectifyKeywordish(WTFMove(string)) } };
    }, [](RefPtr<CSSKeywordValue>&& keywordValue) -> ExceptionOr<RectifiedCSSColorAngle> {
        if (equalIgnoringASCIICase(keywordValue->value(), "none"_s))
            return { WTFMove(keywordValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorAngle"_s };
    });
}

// https://drafts.css-houdini.org/css-typed-om/#rectify-a-csscolornumber
ExceptionOr<RectifiedCSSColorNumber> CSSOMColorValue::rectifyCSSColorNumber(CSSColorNumber&& colorNumber)
{
    return switchOn(WTFMove(colorNumber), [](double value) -> ExceptionOr<RectifiedCSSColorNumber> {
        return { RefPtr<CSSNumericValue> { CSSUnitValue::create(value, CSSUnitType::CSS_NUMBER) } };
    }, [](RefPtr<CSSNumericValue>&& numericValue) -> ExceptionOr<RectifiedCSSColorNumber> {
        if (numericValue->type().matchesNumber())
            return { WTFMove(numericValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorNumber"_s };
    }, [](String&& string) -> ExceptionOr<RectifiedCSSColorNumber> {
        return { RefPtr<CSSKeywordValue> { CSSKeywordValue::rectifyKeywordish(WTFMove(string)) } };
    }, [](RefPtr<CSSKeywordValue>&& keywordValue) -> ExceptionOr<RectifiedCSSColorNumber> {
        if (equalIgnoringASCIICase(keywordValue->value(), "none"_s))
            return { WTFMove(keywordValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorNumber"_s };
    });
}

CSSColorPercent CSSOMColorValue::toCSSColorPercent(const RectifiedCSSColorPercent& component)
{
    return switchOn(component, [](const RefPtr<CSSKeywordValue>& keywordValue) -> CSSColorPercent {
        return keywordValue;
    }, [](const RefPtr<CSSNumericValue>& numericValue) -> CSSColorPercent {
        return numericValue;
    });
}

CSSColorPercent CSSOMColorValue::toCSSColorPercent(const CSSNumberish& numberish)
{
    return switchOn(numberish, [](double number) -> CSSColorPercent {
        return number;
    }, [](const RefPtr<CSSNumericValue>& numericValue) -> CSSColorPercent {
        return numericValue;
    });
}

CSSColorAngle CSSOMColorValue::toCSSColorAngle(const RectifiedCSSColorAngle& angle)
{
    return switchOn(angle, [](const RefPtr<CSSKeywordValue>& keywordValue) -> CSSColorAngle {
        return keywordValue;
    }, [](const RefPtr<CSSNumericValue>& numericValue) -> CSSColorAngle {
        return numericValue;
    });
}

CSSColorNumber CSSOMColorValue::toCSSColorNumber(const RectifiedCSSColorNumber& number)
{
    return switchOn(number, [](const RefPtr<CSSKeywordValue>& keywordValue) -> CSSColorNumber {
        return keywordValue;
    }, [](const RefPtr<CSSNumericValue>& numericValue) -> CSSColorNumber {
        return numericValue;
    });
}

RefPtr<CSSValue> CSSOMColorValue::toCSSValue() const
{
    // FIXME: Implement this.
    return nullptr;
}

} // namespace WebCore
