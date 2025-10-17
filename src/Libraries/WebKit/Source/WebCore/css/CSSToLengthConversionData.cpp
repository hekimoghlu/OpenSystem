/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#include "CSSToLengthConversionData.h"

#include "FloatSize.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include "StyleBuilderState.h"

namespace WebCore {

CSSToLengthConversionData::CSSToLengthConversionData(const RenderStyle& style, Style::BuilderState& builderState)
    : m_style(&style)
    , m_rootStyle(builderState.rootElementStyle())
    , m_parentStyle(&builderState.parentStyle())
    , m_renderView(builderState.document().renderView())
    , m_elementForContainerUnitResolution(builderState.element())
    , m_viewportDependencyDetectionStyle(const_cast<RenderStyle*>(m_style))
    , m_styleBuilderState(&builderState)
{
}

CSSToLengthConversionData::CSSToLengthConversionData(const RenderStyle& style, const RenderStyle* rootStyle, const RenderStyle* parentStyle, const RenderView* renderView, const Element* elementForContainerUnitResolution)
    : m_style(&style)
    , m_rootStyle(rootStyle)
    , m_parentStyle(parentStyle)
    , m_renderView(renderView)
    , m_elementForContainerUnitResolution(elementForContainerUnitResolution)
    , m_zoom(1.f)
    , m_viewportDependencyDetectionStyle(const_cast<RenderStyle*>(m_style))
{
}

const FontCascade& CSSToLengthConversionData::fontCascadeForFontUnits() const
{
    if (computingFontSize()) {
        ASSERT(parentStyle());
        return parentStyle()->fontCascade();
    }
    ASSERT(style());
    return style()->fontCascade();
}

int CSSToLengthConversionData::computedLineHeightForFontUnits() const
{
    if (computingFontSize()) {
        ASSERT(parentStyle());
        return parentStyle()->computedLineHeight();
    }
    ASSERT(style());
    return style()->computedLineHeight();
}

float CSSToLengthConversionData::zoom() const
{
    return m_zoom.value_or(m_style ? m_style->usedZoom() : 1.f);
}

FloatSize CSSToLengthConversionData::defaultViewportFactor() const
{
    if (m_viewportDependencyDetectionStyle)
        m_viewportDependencyDetectionStyle->setUsesViewportUnits();

    if (!m_renderView)
        return { };

    return m_renderView->sizeForCSSDefaultViewportUnits() / 100.0;
}

FloatSize CSSToLengthConversionData::smallViewportFactor() const
{
    if (m_viewportDependencyDetectionStyle)
        m_viewportDependencyDetectionStyle->setUsesViewportUnits();

    if (!m_renderView)
        return { };

    return m_renderView->sizeForCSSSmallViewportUnits() / 100.0;
}

FloatSize CSSToLengthConversionData::largeViewportFactor() const
{
    if (m_viewportDependencyDetectionStyle)
        m_viewportDependencyDetectionStyle->setUsesViewportUnits();

    if (!m_renderView)
        return { };

    return m_renderView->sizeForCSSLargeViewportUnits() / 100.0;
}

FloatSize CSSToLengthConversionData::dynamicViewportFactor() const
{
    if (m_viewportDependencyDetectionStyle)
        m_viewportDependencyDetectionStyle->setUsesViewportUnits();

    if (!m_renderView)
        return { };

    return m_renderView->sizeForCSSDynamicViewportUnits() / 100.0;
}

void CSSToLengthConversionData::setUsesContainerUnits() const
{
    if (m_viewportDependencyDetectionStyle)
        m_viewportDependencyDetectionStyle->setUsesContainerUnits();
}

} // namespace WebCore
