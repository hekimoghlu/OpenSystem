/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

#include "CanvasGradient.h"
#include "CanvasPattern.h"
#include "Color.h"
#include "ColorSerialization.h"
#include <optional>
#include <variant>

namespace WebCore {

class CanvasBase;
class Document;
class GraphicsContext;
class ScriptExecutionContext;

class CanvasStyle {
public:
    CanvasStyle(Color);
    CanvasStyle(const SRGBA<float>&);
    CanvasStyle(CanvasGradient&);
    CanvasStyle(CanvasPattern&);

    static std::optional<CanvasStyle> createFromString(const String& color, CanvasBase&);
    static std::optional<CanvasStyle> createFromStringWithOverrideAlpha(const String& color, float alpha, CanvasBase&);

    String color() const;
    RefPtr<CanvasGradient> canvasGradient() const;
    RefPtr<CanvasPattern> canvasPattern() const;

    void applyFillColor(GraphicsContext&) const;
    void applyStrokeColor(GraphicsContext&) const;

    bool isEquivalentColor(const CanvasStyle&) const;
    bool isEquivalent(const SRGBA<float>&) const;

    template<typename... F>
    decltype(auto) visit(F&&... f) const
    {
        auto visitor = WTF::makeVisitor(std::forward<F>(f)...);
        return WTF::switchOn(m_style,
            [&](const Color& color) {
                return visitor(serializationForHTML(color));
            },
            [&](const Ref<CanvasGradient>& gradient) {
                return visitor(gradient);
            },
            [&](const Ref<CanvasPattern>& pattern) {
                return visitor(pattern);
            }
        );
    }

private:
    std::variant<Color, Ref<CanvasGradient>, Ref<CanvasPattern>> m_style;
};

Color parseColor(const String& colorString, CanvasBase&);
Color parseColor(const String& colorString, ScriptExecutionContext&);

inline RefPtr<CanvasGradient> CanvasStyle::canvasGradient() const
{
    if (!std::holds_alternative<Ref<CanvasGradient>>(m_style))
        return nullptr;
    return std::get<Ref<CanvasGradient>>(m_style).ptr();
}

inline RefPtr<CanvasPattern> CanvasStyle::canvasPattern() const
{
    if (!std::holds_alternative<Ref<CanvasPattern>>(m_style))
        return nullptr;
    return std::get<Ref<CanvasPattern>>(m_style).ptr();
}

inline String CanvasStyle::color() const
{
    if (!std::holds_alternative<Color>(m_style))
        return String();
    return serializationForHTML(std::get<Color>(m_style));
}

} // namespace WebCore
