/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#include "CSSCursorImageValue.h"

#include "CSSImageSetValue.h"
#include "CSSImageValue.h"
#include "SVGCursorElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGLengthContext.h"
#include "SVGURIReference.h"
#include "StyleBuilderState.h"
#include "StyleCursorImage.h"
#include <wtf/MathExtras.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

Ref<CSSCursorImageValue> CSSCursorImageValue::create(Ref<CSSValue>&& value, RefPtr<CSSValue>&& hotSpot, LoadedFromOpaqueSource loadedFromOpaqueSource)
{
    auto* imageValue = dynamicDowncast<CSSImageValue>(value.get());
    auto originalURL = imageValue ? imageValue->imageURL() : URL();
    return adoptRef(*new CSSCursorImageValue(WTFMove(value), WTFMove(hotSpot), WTFMove(originalURL), loadedFromOpaqueSource));
}

Ref<CSSCursorImageValue> CSSCursorImageValue::create(Ref<CSSValue>&& imageValue, RefPtr<CSSValue>&& hotSpot, URL originalURL, LoadedFromOpaqueSource loadedFromOpaqueSource)
{
    return adoptRef(*new CSSCursorImageValue(WTFMove(imageValue), WTFMove(hotSpot), WTFMove(originalURL), loadedFromOpaqueSource));
}

CSSCursorImageValue::CSSCursorImageValue(Ref<CSSValue>&& imageValue, RefPtr<CSSValue>&& hotSpot, URL originalURL, LoadedFromOpaqueSource loadedFromOpaqueSource)
    : CSSValue(ClassType::CursorImage)
    , m_originalURL(WTFMove(originalURL))
    , m_imageValue(WTFMove(imageValue))
    , m_hotSpot(WTFMove(hotSpot))
    , m_loadedFromOpaqueSource(loadedFromOpaqueSource)
{
}

CSSCursorImageValue::~CSSCursorImageValue() = default;

String CSSCursorImageValue::customCSSText() const
{
    auto text = m_imageValue->cssText();
    if (!m_hotSpot)
        return text;
    return makeString(text, ' ', m_hotSpot->first().cssText(), ' ', m_hotSpot->second().cssText());
}

bool CSSCursorImageValue::equals(const CSSCursorImageValue& other) const
{
    return compareCSSValue(m_imageValue, other.m_imageValue)
        && compareCSSValuePtr(m_hotSpot, other.m_hotSpot);
}

RefPtr<StyleCursorImage> CSSCursorImageValue::createStyleImage(const Style::BuilderState& state) const
{
    auto styleImage = state.createStyleImage(m_imageValue.get());
    if (!styleImage)
        return nullptr;

    std::optional<IntPoint> hotSpot;
    if (m_hotSpot) {
        // FIXME: Should we clamp or round instead of just casting from double to int?
        hotSpot = IntPoint {
            static_cast<int>(downcast<CSSPrimitiveValue>(m_hotSpot->first()).resolveAsNumber(state.cssToLengthConversionData())),
            static_cast<int>(downcast<CSSPrimitiveValue>(m_hotSpot->second()).resolveAsNumber(state.cssToLengthConversionData()))
        };
    }
    return StyleCursorImage::create(styleImage.releaseNonNull(), hotSpot, m_originalURL, m_loadedFromOpaqueSource);
}

} // namespace WebCore
