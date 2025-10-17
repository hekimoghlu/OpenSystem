/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#include "CSSFilterImageValue.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "StyleBuilderState.h"
#include "StyleFilterImage.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

CSSFilterImageValue::CSSFilterImageValue(Ref<CSSValue>&& imageValueOrNone, CSS::FilterProperty&& filter)
    : CSSValue { ClassType::FilterImage }
    , m_imageValueOrNone { WTFMove(imageValueOrNone) }
    , m_filter { WTFMove(filter) }
{
}

CSSFilterImageValue::~CSSFilterImageValue() = default;

bool CSSFilterImageValue::equals(const CSSFilterImageValue& other) const
{
    return equalInputImages(other) && m_filter == other.m_filter;
}

bool CSSFilterImageValue::equalInputImages(const CSSFilterImageValue& other) const
{
    return compareCSSValue(m_imageValueOrNone, other.m_imageValueOrNone);
}

String CSSFilterImageValue::customCSSText() const
{
    return makeString("filter("_s, m_imageValueOrNone->cssText(), ", "_s, CSS::serializationForCSS(m_filter), ')');
}

IterationStatus CSSFilterImageValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    if (func(m_imageValueOrNone.get()) == IterationStatus::Done)
        return IterationStatus::Done;
    if (CSS::visitCSSValueChildren(func, m_filter) == IterationStatus::Done)
        return IterationStatus::Done;
    return IterationStatus::Continue;
}

RefPtr<StyleImage> CSSFilterImageValue::createStyleImage(const Style::BuilderState& state) const
{
    return StyleFilterImage::create(state.createStyleImage(m_imageValueOrNone), state.createFilterOperations(m_filter));
}

} // namespace WebCore
