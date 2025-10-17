/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#include "CSSRGB.h"

#include "CSSUnitValue.h"
#include "CSSUnits.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSRGB);

static CSSColorRGBComp toCSSColorRGBComp(const RectifiedCSSColorRGBComp& component)
{
    return switchOn(component, [](const RefPtr<CSSKeywordValue>& keywordValue) -> CSSColorRGBComp {
        return keywordValue;
    }, [](const RefPtr<CSSNumericValue>& numericValue) -> CSSColorRGBComp {
        return numericValue;
    });
}

ExceptionOr<Ref<CSSRGB>> CSSRGB::create(CSSColorRGBComp&& red, CSSColorRGBComp&& green, CSSColorRGBComp&& blue, CSSColorPercent&& alpha)
{
    auto rectifiedRed = rectifyCSSColorRGBComp(WTFMove(red));
    if (rectifiedRed.hasException())
        return rectifiedRed.releaseException();
    auto rectifiedGreen = rectifyCSSColorRGBComp(WTFMove(green));
    if (rectifiedGreen.hasException())
        return rectifiedGreen.releaseException();
    auto rectifiedBlue = rectifyCSSColorRGBComp(WTFMove(blue));
    if (rectifiedBlue.hasException())
        return rectifiedBlue.releaseException();
    auto rectifiedAlpha = rectifyCSSColorPercent(WTFMove(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();

    return adoptRef(*new CSSRGB(rectifiedRed.releaseReturnValue(), rectifiedGreen.releaseReturnValue(), rectifiedBlue.releaseReturnValue(), rectifiedAlpha.releaseReturnValue()));
}

CSSRGB::CSSRGB(RectifiedCSSColorRGBComp&& red, RectifiedCSSColorRGBComp&& green, RectifiedCSSColorRGBComp&& blue, RectifiedCSSColorPercent&& alpha)
    : m_red(WTFMove(red))
    , m_green(WTFMove(green))
    , m_blue(WTFMove(blue))
    , m_alpha(WTFMove(alpha))
{
}

CSSColorRGBComp CSSRGB::r() const
{
    return toCSSColorRGBComp(m_red);
}

ExceptionOr<void> CSSRGB::setR(CSSColorRGBComp&& red)
{
    auto rectifiedRed = rectifyCSSColorRGBComp(WTFMove(red));
    if (rectifiedRed.hasException())
        return rectifiedRed.releaseException();
    m_red = rectifiedRed.releaseReturnValue();
    return { };
}

CSSColorRGBComp CSSRGB::g() const
{
    return toCSSColorRGBComp(m_green);
}

ExceptionOr<void> CSSRGB::setG(CSSColorRGBComp&& green)
{
    auto rectifiedGreen = rectifyCSSColorRGBComp(WTFMove(green));
    if (rectifiedGreen.hasException())
        return rectifiedGreen.releaseException();
    m_green = rectifiedGreen.releaseReturnValue();
    return { };
}

CSSColorRGBComp CSSRGB::b() const
{
    return toCSSColorRGBComp(m_blue);
}

ExceptionOr<void> CSSRGB::setB(CSSColorRGBComp&& blue)
{
    auto rectifiedBlue = rectifyCSSColorRGBComp(WTFMove(blue));
    if (rectifiedBlue.hasException())
        return rectifiedBlue.releaseException();
    m_blue = rectifiedBlue.releaseReturnValue();
    return { };
}

CSSColorPercent CSSRGB::alpha() const
{
    return toCSSColorPercent(m_alpha);
}

ExceptionOr<void> CSSRGB::setAlpha(CSSColorPercent&& alpha)
{
    auto rectifiedAlpha = rectifyCSSColorPercent(WTFMove(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();
    m_alpha = rectifiedAlpha.releaseReturnValue();
    return { };
}

// https://drafts.css-houdini.org/css-typed-om-1/#rectify-a-csscolorrgbcomp
ExceptionOr<RectifiedCSSColorRGBComp> CSSRGB::rectifyCSSColorRGBComp(CSSColorRGBComp&& component)
{
    return switchOn(WTFMove(component), [](double value) -> ExceptionOr<RectifiedCSSColorRGBComp> {
        return { RefPtr<CSSNumericValue> { CSSUnitValue::create(value * 100, CSSUnitType::CSS_PERCENTAGE) } };
    }, [](RefPtr<CSSNumericValue>&& numericValue) -> ExceptionOr<RectifiedCSSColorRGBComp> {
        if (numericValue->type().matchesNumber() || numericValue->type().matches<CSSNumericBaseType::Percent>())
            return { WTFMove(numericValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorRGBComp"_s };
    }, [](String&& string) -> ExceptionOr<RectifiedCSSColorRGBComp> {
        return { RefPtr<CSSKeywordValue> { CSSKeywordValue::rectifyKeywordish(WTFMove(string)) } };
    }, [](RefPtr<CSSKeywordValue>&& keywordValue) -> ExceptionOr<RectifiedCSSColorRGBComp> {
        if (equalIgnoringASCIICase(keywordValue->value(), "none"_s))
            return { WTFMove(keywordValue) };
        return Exception { ExceptionCode::SyntaxError, "Invalid CSSColorRGBComp"_s };
    });
}

} // namespace WebCore
