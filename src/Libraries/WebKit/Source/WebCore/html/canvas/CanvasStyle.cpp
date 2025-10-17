/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#include "CanvasStyle.h"

#include "CSSParser.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParserConsumer+Color.h"
#include "CanvasGradient.h"
#include "CanvasPattern.h"
#include "ColorConversion.h"
#include "Gradient.h"
#include "GraphicsContext.h"
#include "HTMLCanvasElement.h"
#include "StyleProperties.h"

#if ENABLE(OFFSCREEN_CANVAS)
#include "OffscreenCanvas.h"
#endif

namespace WebCore {

class CanvasStyleColorResolutionDelegate final : public CSS::PlatformColorResolutionDelegate {
public:
    explicit CanvasStyleColorResolutionDelegate(Ref<HTMLCanvasElement> canvasElement)
        : m_canvasElement { WTFMove(canvasElement) }
    {
    }

    Color currentColor() const final;

    Ref<HTMLCanvasElement> m_canvasElement;
};

using LazySlowPathColorParsingParameters = std::tuple<
    CSSPropertyParserHelpers::CSSColorParsingOptions,
    CSS::PlatformColorResolutionState,
    std::optional<CanvasStyleColorResolutionDelegate>
>;

Color CanvasStyleColorResolutionDelegate::currentColor() const
{
    if (!m_canvasElement->isConnected() || !m_canvasElement->inlineStyle())
        return Color::black;

    auto colorString = m_canvasElement->inlineStyle()->getPropertyValue(CSSPropertyColor);
    auto color = CSSPropertyParserHelpers::parseColorRaw(WTFMove(colorString), m_canvasElement->cssParserContext(), [] {
        return LazySlowPathColorParsingParameters { { }, { }, std::nullopt };
    });
    if (!color.isValid())
        return Color::black;
    return color;
}

static OptionSet<CSS::ColorType> allowedColorTypes(ScriptExecutionContext* scriptExecutionContext)
{
    if (scriptExecutionContext && scriptExecutionContext->isDocument())
        return { CSS::ColorType::Absolute, CSS::ColorType::Current, CSS::ColorType::System };

    // FIXME: All canvas types should support all color types, but currently
    //        system colors are not thread safe so are disabled for non-document
    //        based canvases.
    return { CSS::ColorType::Absolute, CSS::ColorType::Current };
}

static LazySlowPathColorParsingParameters elementlessColorParsingParameters(ScriptExecutionContext* scriptExecutionContext)
{
    return {
        CSSPropertyParserHelpers::CSSColorParsingOptions {
            .allowedColorTypes = allowedColorTypes(scriptExecutionContext)
        },
        CSS::PlatformColorResolutionState {
            .resolvedCurrentColor = Color::black
        },
        std::nullopt
    };
}

static LazySlowPathColorParsingParameters colorParsingParameters(CanvasBase& canvasBase)
{
    RefPtr canvasElement = dynamicDowncast<HTMLCanvasElement>(canvasBase);
    if (!canvasElement)
        return elementlessColorParsingParameters(canvasBase.scriptExecutionContext());

    return {
        CSSPropertyParserHelpers::CSSColorParsingOptions { },
        CSS::PlatformColorResolutionState { },
        CanvasStyleColorResolutionDelegate(canvasElement.releaseNonNull())
    };
}

Color parseColor(const String& colorString, CanvasBase& canvasBase)
{
    return CSSPropertyParserHelpers::parseColorRaw(colorString, canvasBase.cssParserContext(), [&] {
        return colorParsingParameters(canvasBase);
    });
}

Color parseColor(const String& colorString, ScriptExecutionContext& scriptExecutionContext)
{
    // FIXME: Add constructor for CSSParserContext that takes a ScriptExecutionContext to allow preferences to be
    //        checked correctly.

    return CSSPropertyParserHelpers::parseColorRaw(colorString, CSSParserContext(HTMLStandardMode), [&] {
        return elementlessColorParsingParameters(&scriptExecutionContext);
    });
}

CanvasStyle::CanvasStyle(Color color)
    : m_style(color)
{
}

CanvasStyle::CanvasStyle(const SRGBA<float>& colorComponents)
    : m_style(convertColor<SRGBA<uint8_t>>(colorComponents))
{
}

CanvasStyle::CanvasStyle(CanvasGradient& gradient)
    : m_style(gradient)
{
}

CanvasStyle::CanvasStyle(CanvasPattern& pattern)
    : m_style(pattern)
{
}

std::optional<CanvasStyle> CanvasStyle::createFromString(const String& colorString, CanvasBase& canvasBase)
{
    auto color = parseColor(colorString, canvasBase);
    if (!color.isValid())
        return { };

    return { color };
}

std::optional<CanvasStyle> CanvasStyle::createFromStringWithOverrideAlpha(const String& colorString, float alpha, CanvasBase& canvasBase)
{
    auto color = parseColor(colorString, canvasBase);
    if (!color.isValid())
        return { };

    return { color.colorWithAlpha(alpha) };
}

bool CanvasStyle::isEquivalentColor(const CanvasStyle& other) const
{
    if (std::holds_alternative<Color>(m_style) && std::holds_alternative<Color>(other.m_style))
        return std::get<Color>(m_style) == std::get<Color>(other.m_style);

    return false;
}

bool CanvasStyle::isEquivalent(const SRGBA<float>& components) const
{
    return std::holds_alternative<Color>(m_style) && std::get<Color>(m_style) == convertColor<SRGBA<uint8_t>>(components);
}

void CanvasStyle::applyStrokeColor(GraphicsContext& context) const
{
    WTF::switchOn(m_style,
        [&context](const Color& color) {
            context.setStrokeColor(color);
        },
        [&context](const Ref<CanvasGradient>& gradient) {
            context.setStrokeGradient(gradient->gradient());
        },
        [&context](const Ref<CanvasPattern>& pattern) {
            context.setStrokePattern(pattern->pattern());
        }
    );
}

void CanvasStyle::applyFillColor(GraphicsContext& context) const
{
    WTF::switchOn(m_style,
        [&context](const Color& color) {
            context.setFillColor(color);
        },
        [&context](const Ref<CanvasGradient>& gradient) {
            context.setFillGradient(gradient->gradient());
        },
        [&context](const Ref<CanvasPattern>& pattern) {
            context.setFillPattern(pattern->pattern());
        }
    );
}

}
