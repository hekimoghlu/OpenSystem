/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#include "CSSCrossfadeValue.h"

#include "StyleBuilderState.h"
#include "StyleCrossfadeImage.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

inline CSSCrossfadeValue::CSSCrossfadeValue(Ref<CSSValue>&& fromValueOrNone, Ref<CSSValue>&& toValueOrNone, Ref<CSSPrimitiveValue>&& percentageValue, bool isPrefixed)
    : CSSValue { ClassType::Crossfade }
    , m_fromValueOrNone { WTFMove(fromValueOrNone) }
    , m_toValueOrNone { WTFMove(toValueOrNone) }
    , m_percentageValue { WTFMove(percentageValue) }
    , m_isPrefixed { isPrefixed }
{
}

Ref<CSSCrossfadeValue> CSSCrossfadeValue::create(Ref<CSSValue>&& fromValueOrNone, Ref<CSSValue>&& toValueOrNone, Ref<CSSPrimitiveValue>&& percentageValue, bool isPrefixed)
{
    return adoptRef(*new CSSCrossfadeValue(WTFMove(fromValueOrNone), WTFMove(toValueOrNone), WTFMove(percentageValue), isPrefixed));
}

CSSCrossfadeValue::~CSSCrossfadeValue() = default;

bool CSSCrossfadeValue::equals(const CSSCrossfadeValue& other) const
{
    return equalInputImages(other) && compareCSSValue(m_percentageValue, other.m_percentageValue);
}

bool CSSCrossfadeValue::equalInputImages(const CSSCrossfadeValue& other) const
{
    return compareCSSValue(m_fromValueOrNone, other.m_fromValueOrNone) && compareCSSValue(m_toValueOrNone, other.m_toValueOrNone);
}

String CSSCrossfadeValue::customCSSText() const
{
    return makeString(m_isPrefixed ? "-webkit-"_s : ""_s, "cross-fade("_s, m_fromValueOrNone->cssText(), ", "_s, m_toValueOrNone->cssText(), ", "_s, m_percentageValue->cssText(), ')');
}

RefPtr<StyleImage> CSSCrossfadeValue::createStyleImage(const Style::BuilderState& state) const
{
    return StyleCrossfadeImage::create(
        state.createStyleImage(m_fromValueOrNone),
        state.createStyleImage(m_toValueOrNone),
        clampTo<double>(m_percentageValue->valueDividingBy100IfPercentage<double>(state.cssToLengthConversionData()), 0, 1),
        m_isPrefixed
    );
}

} // namespace WebCore
