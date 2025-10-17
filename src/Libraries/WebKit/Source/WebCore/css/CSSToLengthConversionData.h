/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#pragma once

#include "CSSPropertyNames.h"
#include "Element.h"
#include <optional>
#include <wtf/Assertions.h>

namespace WebCore {

class Element;
class FloatSize;
class FontCascade;
class RenderStyle;
class RenderView;

namespace Style {
class BuilderState;
};

class CSSToLengthConversionData {
public:
    // This is used during style building. The 'zoom' property is taken into account.
    CSSToLengthConversionData(const RenderStyle&, Style::BuilderState&);
    // This constructor ignores the `zoom` property.
    CSSToLengthConversionData(const RenderStyle&, const RenderStyle* rootStyle, const RenderStyle* parentStyle, const RenderView*, const Element* elementForContainerUnitResolution = nullptr);

    CSSToLengthConversionData() = default;

    const RenderStyle* style() const { return m_style; }
    const RenderStyle* rootStyle() const { return m_rootStyle; }
    const RenderStyle* parentStyle() const { return m_parentStyle; }
    float zoom() const;
    bool computingFontSize() const { return m_propertyToCompute == CSSPropertyFontSize; }
    bool computingLineHeight() const { return m_propertyToCompute == CSSPropertyLineHeight; }
    CSSPropertyID propertyToCompute() const { return m_propertyToCompute.value_or(CSSPropertyInvalid); }
    const RenderView* renderView() const { return m_renderView; }
    const Element* elementForContainerUnitResolution() const { return m_elementForContainerUnitResolution.get(); }

    const FontCascade& fontCascadeForFontUnits() const;
    int computedLineHeightForFontUnits() const;

    FloatSize defaultViewportFactor() const;
    FloatSize smallViewportFactor() const;
    FloatSize largeViewportFactor() const;
    FloatSize dynamicViewportFactor() const;

    CSSToLengthConversionData copyForFontSize() const
    {
        CSSToLengthConversionData copy(*this);
        copy.m_zoom = 1.f;
        copy.m_propertyToCompute = CSSPropertyFontSize;
        return copy;
    };

    CSSToLengthConversionData copyWithAdjustedZoom(float zoom) const
    {
        CSSToLengthConversionData copy(*this);
        copy.m_zoom = zoom;
        return copy;
    }

    CSSToLengthConversionData copyForLineHeight(float zoom) const
    {
        CSSToLengthConversionData copy(*this);
        copy.m_zoom = zoom;
        copy.m_propertyToCompute = CSSPropertyLineHeight;
        return copy;
    }

    void setUsesContainerUnits() const;

    Style::BuilderState* styleBuilderState() const { return m_styleBuilderState; }

private:
    const RenderStyle* m_style { nullptr };
    const RenderStyle* m_rootStyle { nullptr };
    const RenderStyle* m_parentStyle { nullptr };
    const RenderView* m_renderView { nullptr };
    RefPtr<const Element> m_elementForContainerUnitResolution;
    std::optional<float> m_zoom;
    std::optional<CSSPropertyID> m_propertyToCompute;
    // FIXME: Remove this hack.
    RenderStyle* m_viewportDependencyDetectionStyle { nullptr };

    Style::BuilderState* m_styleBuilderState { nullptr };
};

} // namespace WebCore
