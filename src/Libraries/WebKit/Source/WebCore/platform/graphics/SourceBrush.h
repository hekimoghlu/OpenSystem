/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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

#include "Color.h"
#include "Gradient.h"
#include "Pattern.h"
#include "SourceBrushLogicalGradient.h"

namespace WebCore {

class SourceBrush {
public:
    using OptionalPatternGradient = std::variant<std::monostate, SourceBrushLogicalGradient, Ref<Pattern>>;

    SourceBrush() = default;
    WEBCORE_EXPORT SourceBrush(const Color&, OptionalPatternGradient&& = std::monostate { });

    const Color& color() const { return m_color; }
    void setColor(const Color color) { m_color = color; }

    const OptionalPatternGradient& patternGradient() const { return m_patternGradient; }

    WEBCORE_EXPORT Gradient* gradient() const;
    WEBCORE_EXPORT Pattern* pattern() const;
    RefPtr<Pattern> protectedPattern() const { return pattern(); }
    WEBCORE_EXPORT const AffineTransform& gradientSpaceTransform() const;
    WEBCORE_EXPORT std::optional<RenderingResourceIdentifier> gradientIdentifier() const;

    WEBCORE_EXPORT void setGradient(Ref<Gradient>&&, const AffineTransform& spaceTransform = { });
    void setPattern(Ref<Pattern>&&);

    bool isInlineColor() const { return !hasPatternOrGradient() && m_color.tryGetAsSRGBABytes(); }
    bool isVisible() const { return hasPatternOrGradient() || m_color.isVisible(); }

    bool hasPatternOrGradient() const { return !std::holds_alternative<std::monostate>(m_patternGradient); }
    friend bool operator==(const SourceBrush&, const SourceBrush&);

private:
    Color m_color { Color::black };
    OptionalPatternGradient m_patternGradient;
};

inline bool operator==(const SourceBrush& a, const SourceBrush& b)
{
    // Workaround for Ref<> lack of operator==.
    auto patternGradientEqual = [](const SourceBrush::OptionalPatternGradient& a, const SourceBrush::OptionalPatternGradient& b) -> bool {
        if (a.index() != b.index())
            return false;
        if (auto* aGradient = std::get_if<SourceBrushLogicalGradient>(&a))
            return *aGradient == std::get<SourceBrushLogicalGradient>(b);
        if (auto* aPattern = std::get_if<Ref<Pattern>>(&a))
            return aPattern->ptr() == std::get<Ref<Pattern>>(b).ptr();
        return true;
    };
    return a.m_color == b.m_color && patternGradientEqual(a.m_patternGradient, b.m_patternGradient);
}

WTF::TextStream& operator<<(WTF::TextStream&, const SourceBrush&);

} // namespace WebCore
