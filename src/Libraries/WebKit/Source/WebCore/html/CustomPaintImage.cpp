/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#include "CustomPaintImage.h"

#include "CSSComputedStyleDeclaration.h"
#include "CSSImageValue.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParser.h"
#include "CSSStyleImageValue.h"
#include "CSSUnitValue.h"
#include "CSSUnparsedValue.h"
#include "CustomPaintCanvas.h"
#include "GraphicsContext.h"
#include "HashMapStylePropertyMapReadOnly.h"
#include "ImageBitmap.h"
#include "ImageBuffer.h"
#include "JSCSSPaintCallback.h"
#include "JSDOMExceptionHandling.h"
#include "MainThreadStylePropertyMapReadOnly.h"
#include "PaintRenderingContext2D.h"
#include "RenderElement.h"
#include <JavaScriptCore/ConstructData.h>

namespace WebCore {

CustomPaintImage::CustomPaintImage(PaintDefinition& definition, const FloatSize& size, const RenderElement& element, const Vector<String>& arguments)
    : m_paintDefinition(definition)
    , m_inputProperties(definition.inputProperties)
    , m_element(element)
    , m_arguments(arguments)
{
    setContainerSize(size);
}

CustomPaintImage::~CustomPaintImage() = default;

static RefPtr<CSSValue> extractComputedProperty(const AtomString& name, Element& element)
{
    ComputedStyleExtractor extractor(&element);

    if (isCustomPropertyName(name))
        return extractor.customPropertyValue(name);

    CSSPropertyID propertyID = cssPropertyID(name);
    if (!propertyID)
        return nullptr;

    return extractor.propertyValue(propertyID, ComputedStyleExtractor::UpdateLayout::No);
}

ImageDrawResult CustomPaintImage::doCustomPaint(GraphicsContext& destContext, const FloatSize& destSize)
{
    if (!m_element || !m_element->element() || !m_paintDefinition)
        return ImageDrawResult::DidNothing;

    JSC::JSValue paintConstructor = m_paintDefinition->paintConstructor;

    if (!paintConstructor)
        return ImageDrawResult::DidNothing;

    ASSERT(!m_element->needsLayout());
    ASSERT(!m_element->element()->document().needsStyleRecalc());

    Ref callback = static_cast<JSCSSPaintCallback&>(m_paintDefinition->paintCallback.get());
    RefPtr scriptExecutionContext = callback->scriptExecutionContext();
    if (!scriptExecutionContext)
        return ImageDrawResult::DidNothing;

    Ref canvas = CustomPaintCanvas::create(*scriptExecutionContext, destSize.width(), destSize.height());
    RefPtr context = canvas->getContext();

    UncheckedKeyHashMap<AtomString, RefPtr<CSSValue>> propertyValues;

    if (auto* element = m_element->element()) {
        for (auto& name : m_inputProperties)
            propertyValues.add(name, extractComputedProperty(name, *element));
    }

    auto size = CSSPaintSize::create(destSize.width(), destSize.height());
    Ref<StylePropertyMapReadOnly> propertyMap = HashMapStylePropertyMapReadOnly::create(WTFMove(propertyValues));

    auto& vm = paintConstructor.getObject()->vm();
    JSC::JSLockHolder lock(vm);
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto& globalObject = *paintConstructor.getObject()->globalObject();

    auto& lexicalGlobalObject = globalObject;
    JSC::ArgList noArgs;
    JSC::JSValue thisObject = { JSC::construct(&lexicalGlobalObject, paintConstructor, noArgs, "Failed to construct paint class"_s) };

    if (UNLIKELY(scope.exception())) {
        reportException(&lexicalGlobalObject, scope.exception());
        return ImageDrawResult::DidNothing;
    }

    auto result = callback->handleEvent(WTFMove(thisObject), *context, size, propertyMap, m_arguments);
    if (result.type() != CallbackResultType::Success)
        return ImageDrawResult::DidNothing;

    canvas->replayDisplayList(destContext);

    return ImageDrawResult::DidDraw;
}

ImageDrawResult CustomPaintImage::draw(GraphicsContext& destContext, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions options)
{
    GraphicsContextStateSaver stateSaver(destContext);
    destContext.setCompositeOperation(options.compositeOperator(), options.blendMode());
    destContext.clip(destRect);
    destContext.translate(destRect.location());
    if (destRect.size() != srcRect.size())
        destContext.scale(destRect.size() / srcRect.size());
    destContext.translate(-srcRect.location());
    return doCustomPaint(destContext, size());
}

void CustomPaintImage::drawPattern(GraphicsContext& destContext, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform,
    const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    // Allow the generator to provide visually-equivalent tiling parameters for better performance.
    FloatSize adjustedSize = size();
    FloatRect adjustedSrcRect = srcRect;

    // Factor in the destination context's scale to generate at the best resolution
    AffineTransform destContextCTM = destContext.getCTM(GraphicsContext::DefinitelyIncludeDeviceScale);
    double xScale = std::abs(destContextCTM.xScale());
    double yScale = std::abs(destContextCTM.yScale());
    AffineTransform adjustedPatternCTM = patternTransform;
    adjustedPatternCTM.scale(1.0 / xScale, 1.0 / yScale);
    adjustedSrcRect.scale(xScale, yScale);

    auto buffer = destContext.createAlignedImageBuffer(adjustedSize);
    if (!buffer)
        return;
    doCustomPaint(buffer->context(), adjustedSize);

    if (destContext.drawLuminanceMask())
        buffer->convertToLuminanceMask();

    destContext.drawPattern(*buffer, destRect, adjustedSrcRect, adjustedPatternCTM, phase, spacing, options);
    destContext.setDrawLuminanceMask(false);
}

} // namespace WebCore
