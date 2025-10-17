/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#include "OffscreenCanvasRenderingContext2D.h"

#if ENABLE(OFFSCREEN_CANVAS)

#include "CSSFontSelector.h"
#include "CSSParserContext.h"
#include "CSSPropertyParserConsumer+Font.h"
#include "InspectorInstrumentation.h"
#include "RenderStyle.h"
#include "ScriptExecutionContext.h"
#include "StyleResolveForFont.h"
#include "TextMetrics.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(OffscreenCanvasRenderingContext2D);

bool OffscreenCanvasRenderingContext2D::enabledForContext(ScriptExecutionContext& context)
{
    UNUSED_PARAM(context);
#if ENABLE(OFFSCREEN_CANVAS_IN_WORKERS)
    if (context.isWorkerGlobalScope())
        return context.settingsValues().offscreenCanvasInWorkersEnabled;
#endif

    ASSERT(context.isDocument());
    return true;
}


std::unique_ptr<OffscreenCanvasRenderingContext2D> OffscreenCanvasRenderingContext2D::create(CanvasBase& canvas, CanvasRenderingContext2DSettings&& settings)
{
    auto renderingContext = std::unique_ptr<OffscreenCanvasRenderingContext2D>(new OffscreenCanvasRenderingContext2D(canvas, WTFMove(settings)));

    InspectorInstrumentation::didCreateCanvasRenderingContext(*renderingContext);

    return renderingContext;
}

OffscreenCanvasRenderingContext2D::OffscreenCanvasRenderingContext2D(CanvasBase& canvas, CanvasRenderingContext2DSettings&& settings)
    : CanvasRenderingContext2DBase(canvas, Type::Offscreen2D, WTFMove(settings), false)
{
}

OffscreenCanvasRenderingContext2D::~OffscreenCanvasRenderingContext2D() = default;

void OffscreenCanvasRenderingContext2D::setFont(const String& newFont)
{
    auto& context = *canvasBase().scriptExecutionContext();

    if (newFont.isEmpty())
        return;

    if (newFont == state().unparsedFont && state().font.realized())
        return;

    // According to http://lists.w3.org/Archives/Public/public-html/2009Jul/0947.html,
    // the "inherit" and "initial" values must be ignored. CSSPropertyParserHelpers::parseFont() ignores these.
    auto unresolvedFont = CSSPropertyParserHelpers::parseUnresolvedFont(newFont, strictToCSSParserMode(!usesCSSCompatibilityParseMode()));
    if (!unresolvedFont)
        return;

    // The parse succeeded.
    String newFontSafeCopy(newFont); // Create a string copy since newFont can be deleted inside realizeSaves.
    realizeSaves();
    modifiableState().unparsedFont = newFontSafeCopy;

    // Map the <canvas> font into the text style. If the font uses keywords like larger/smaller, these will work
    // relative to the default font.
    FontCascadeDescription fontDescription;
    fontDescription.setOneFamily(DefaultFontFamily);
    fontDescription.setSpecifiedSize(DefaultFontSize);
    fontDescription.setComputedSize(DefaultFontSize);

    if (auto fontCascade = Style::resolveForUnresolvedFont(*unresolvedFont, WTFMove(fontDescription), context)) {
        ASSERT(context.cssFontSelector());
        modifiableState().font.initialize(*context.cssFontSelector(), *fontCascade);

        String letterSpacing;
        setLetterSpacing(std::exchange(modifiableState().letterSpacing, letterSpacing));
        String wordSpacing;
        setWordSpacing(std::exchange(modifiableState().wordSpacing, wordSpacing));
    }
}

RefPtr<ImageBuffer> OffscreenCanvasRenderingContext2D::transferToImageBuffer()
{
    if (!canvasBase().hasCreatedImageBuffer())
        return canvasBase().allocateImageBuffer();
    auto* buffer = canvasBase().buffer();
    if (!buffer)
        return nullptr;
    // As the canvas context state is stored in GraphicsContext, which is owned
    // by buffer(), to avoid resetting the context state, we have to make a copy and
    // clear the original buffer rather than returning the original buffer.
    RefPtr result = buffer->clone();
    clearCanvas();
    return result;
}

CanvasDirection OffscreenCanvasRenderingContext2D::direction() const
{
    // FIXME: What should we do about inherit here?
    switch (state().direction) {
    case Direction::Inherit:
    case Direction::Ltr:
        return Direction::Ltr;
    case Direction::Rtl:
        return Direction::Rtl;
    }
    ASSERT_NOT_REACHED();
    return Direction::Ltr;
}

auto OffscreenCanvasRenderingContext2D::fontProxy() -> const FontProxy* {
    if (!state().font.realized())
        setFont(state().unparsedFont);
    return &state().font;
}

void OffscreenCanvasRenderingContext2D::fillText(const String& text, double x, double y, std::optional<double> maxWidth)
{
    drawText(text, x, y, true, maxWidth);
}

void OffscreenCanvasRenderingContext2D::strokeText(const String& text, double x, double y, std::optional<double> maxWidth)
{
    drawText(text, x, y, false, maxWidth);
}

Ref<TextMetrics> OffscreenCanvasRenderingContext2D::measureText(const String& text)
{
    return measureTextInternal(text);
}

} // namespace WebCore

#endif
