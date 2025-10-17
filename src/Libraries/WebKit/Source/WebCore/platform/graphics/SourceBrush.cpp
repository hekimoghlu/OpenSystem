/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#include "SourceBrush.h"

namespace WebCore {

SourceBrush::SourceBrush(const Color& color, OptionalPatternGradient&& patternGradient)
    : m_color(color)
    , m_patternGradient(WTFMove(patternGradient))
{
}

const AffineTransform& SourceBrush::gradientSpaceTransform() const
{
    if (auto* logicalGradient = std::get_if<SourceBrushLogicalGradient>(&m_patternGradient))
        return logicalGradient->spaceTransform;
    return identity;
}

Gradient* SourceBrush::gradient() const
{
    if (auto* logicalGradient = std::get_if<SourceBrushLogicalGradient>(&m_patternGradient)) {
        if (auto* gradient = std::get_if<Ref<Gradient>>(&logicalGradient->gradient))
            return gradient->ptr();
    }
    return nullptr;
}

std::optional<RenderingResourceIdentifier> SourceBrush::gradientIdentifier() const
{
    auto* gradient = std::get_if<SourceBrushLogicalGradient>(&m_patternGradient);
    if (!gradient)
        return std::nullopt;

    return WTF::switchOn(gradient->gradient,
        [] (const Ref<Gradient>& gradient) -> std::optional<RenderingResourceIdentifier> {
            if (!gradient->hasValidRenderingResourceIdentifier())
                return std::nullopt;
            return gradient->renderingResourceIdentifier();
        },
        [] (RenderingResourceIdentifier renderingResourceIdentifier) -> std::optional<RenderingResourceIdentifier> {
            return renderingResourceIdentifier;
        }
    );
}

Pattern* SourceBrush::pattern() const
{
    if (auto* pattern = std::get_if<Ref<Pattern>>(&m_patternGradient))
        return pattern->ptr();
    return nullptr;
}

void SourceBrush::setGradient(Ref<Gradient>&& gradient, const AffineTransform& spaceTransform)
{
    m_patternGradient = SourceBrushLogicalGradient { WTFMove(gradient), spaceTransform };
}

void SourceBrush::setPattern(Ref<Pattern>&& pattern)
{
    m_patternGradient.emplace<Ref<Pattern>>(WTFMove(pattern));
}

WTF::TextStream& operator<<(TextStream& ts, const SourceBrush& brush)
{
    ts.dumpProperty("color", brush.color());

    if (auto gradient = brush.gradient()) {
        ts.dumpProperty("gradient", *gradient);
        ts.dumpProperty("gradient-space-transform", brush.gradientSpaceTransform());
    }

    if (auto pattern = brush.pattern())
        ts.dumpProperty("pattern", pattern);

    return ts;
}

} // namespace WebCore
