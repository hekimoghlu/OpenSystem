/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#include "CustomPaintCanvas.h"

#include "BitmapImage.h"
#include "CSSParserContext.h"
#include "CanvasRenderingContext.h"
#include "ImageBitmap.h"
#include "PaintRenderingContext2D.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CustomPaintCanvas);

Ref<CustomPaintCanvas> CustomPaintCanvas::create(ScriptExecutionContext& context, unsigned width, unsigned height)
{
    return adoptRef(*new CustomPaintCanvas(context, width, height));
}

CustomPaintCanvas::CustomPaintCanvas(ScriptExecutionContext& context, unsigned width, unsigned height)
    : CanvasBase(IntSize(width, height), context)
    , ContextDestructionObserver(&context)
{
}

CustomPaintCanvas::~CustomPaintCanvas()
{
    notifyObserversCanvasDestroyed();

    m_context = nullptr; // Ensure this goes away before the ImageBuffer.
    setImageBuffer(nullptr);
}

RefPtr<PaintRenderingContext2D> CustomPaintCanvas::getContext()
{
    if (!m_context)
        m_context = PaintRenderingContext2D::create(*this);
    return m_context.get();
}

void CustomPaintCanvas::replayDisplayList(GraphicsContext& target)
{
    if (!width() || !height())
        return;
    // FIXME: Using an intermediate buffer is not needed if there are no composite operations.
    auto clipBounds = target.clipBounds();
    auto image = target.createAlignedImageBuffer(clipBounds.size());
    if (!image)
        return;
    auto& imageTarget = image->context();
    imageTarget.translate(-clipBounds.location());
    if (m_context)
        m_context->replayDisplayList(imageTarget);
    target.drawImageBuffer(*image, clipBounds);
}

Image* CustomPaintCanvas::copiedImage() const
{
    if (!width() || !height())
        return nullptr;
    m_copiedImage = nullptr;
    auto buffer = ImageBuffer::create(size(), RenderingMode::Unaccelerated, RenderingPurpose::Unspecified, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8);
    if (buffer) {
        if (m_context)
            m_context->replayDisplayList(buffer->context());
        m_copiedImage = BitmapImage::create(ImageBuffer::sinkIntoNativeImage(buffer));
    }
    return m_copiedImage.get();
}

void CustomPaintCanvas::clearCopiedImage() const
{
    m_copiedImage = nullptr;
}

const CSSParserContext& CustomPaintCanvas::cssParserContext() const
{
    // FIXME: Rather than using a default CSSParserContext, there should be one exposed via ScriptExecutionContext.
    if (!m_cssParserContext)
        m_cssParserContext = WTF::makeUnique<CSSParserContext>(HTMLStandardMode);
    return *m_cssParserContext;
}

} // namespace WebCore
