/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#include "ImageBitmapRenderingContext.h"

#include "HTMLCanvasElement.h"
#include "ImageBitmap.h"
#include "ImageBuffer.h"
#include "InspectorInstrumentation.h"
#include "OffscreenCanvas.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBitmapRenderingContext);

std::unique_ptr<ImageBitmapRenderingContext> ImageBitmapRenderingContext::create(CanvasBase& canvas, ImageBitmapRenderingContextSettings&& settings)
{
    auto renderingContext = std::unique_ptr<ImageBitmapRenderingContext>(new ImageBitmapRenderingContext(canvas, WTFMove(settings)));

    InspectorInstrumentation::didCreateCanvasRenderingContext(*renderingContext);

    return renderingContext;
}

ImageBitmapRenderingContext::ImageBitmapRenderingContext(CanvasBase& canvas, ImageBitmapRenderingContextSettings&& settings)
    : CanvasRenderingContext(canvas, Type::BitmapRenderer)
    , m_settings(WTFMove(settings))
{
}

ImageBitmapRenderingContext::~ImageBitmapRenderingContext() = default;

ImageBitmapCanvas ImageBitmapRenderingContext::canvas()
{
    auto& base = canvasBase();
#if ENABLE(OFFSCREEN_CANVAS)
    if (auto* offscreenCanvas = dynamicDowncast<OffscreenCanvas>(base))
        return offscreenCanvas;
#endif
    return &downcast<HTMLCanvasElement>(base);
}

void ImageBitmapRenderingContext::setOutputBitmap(RefPtr<ImageBitmap> imageBitmap)
{
    // 1. If a bitmap argument was not provided, then:

    if (!imageBitmap) {
        // 1.1. Set context's bitmap mode to blank.
        // 1.2. Let canvas be the canvas element to which context is bound.
        // 1.3. Set context's output bitmap to be transparent black with an
        //      intrinsic width equal to the numeric value of canvas's width attribute
        //      and an intrinsic height equal to the numeric value of canvas's height
        //      attribute, those values being interpreted in CSS pixels.
        setBlank();
        // 1.4. Set the output bitmap's origin-clean flag to true.
        canvasBase().setOriginClean();
        return;
    }

    // 2. If a bitmap argument was provided, then:

    // 2.1. Set context's bitmap mode to valid.

    m_bitmapMode = BitmapMode::Valid;

    // 2.2. Set context's output bitmap to refer to the same underlying
    //      bitmap data as bitmap, without making a copy.
    //      Note: the origin-clean flag of bitmap is included in the
    //      bitmap data to be referenced by context's output bitmap.

    if (imageBitmap->originClean())
        canvasBase().setOriginClean();
    else
        canvasBase().setOriginTainted();
    canvasBase().setImageBufferAndMarkDirty(imageBitmap->takeImageBuffer());
}

ExceptionOr<void> ImageBitmapRenderingContext::transferFromImageBitmap(RefPtr<ImageBitmap> imageBitmap)
{
    // 1. Let bitmapContext be the ImageBitmapRenderingContext object on which
    //    the transferFromImageBitmap() method was called.

    // 2. If imageBitmap is null, then run the steps to set an ImageBitmapRenderingContext's
    //    output bitmap, with bitmapContext as the context argument and no bitmap argument,
    //    then abort these steps.

    if (!imageBitmap) {
        setOutputBitmap(nullptr);
        return { };
    }

    // 3. If the value of imageBitmap's [[Detached]] internal slot is set to true,
    //    then throw an "InvalidStateError" DOMException and abort these steps.

    if (imageBitmap->isDetached())
        return Exception { ExceptionCode::InvalidStateError };

    // 4. Run the steps to set an ImageBitmapRenderingContext's output bitmap,
    //    with the context argument equal to bitmapContext, and the bitmap
    //    argument referring to imageBitmap's underlying bitmap data.

    setOutputBitmap(imageBitmap);

    // 5. Set the value of imageBitmap's [[Detached]] internal slot to true.
    // 6. Unset imageBitmap's bitmap data.

    // Note that the algorithm in the specification is currently a bit
    // muddy here. The setOutputBitmap step above had to transfer ownership
    // from the imageBitmap to this object, which requires a detach and unset,
    // so this step isn't necessary, but we'll do it anyway.

    imageBitmap->close();

    return { };
}

void ImageBitmapRenderingContext::setBlank()
{
    m_bitmapMode = BitmapMode::Blank;
    // FIXME: What is the point of creating a full size transparent buffer that
    // can never be changed? Wouldn't a 1x1 buffer give the same rendering? The
    // only reason I can think of is toDataURL(), but that doesn't seem like
    // a good enough argument to waste memory.
    auto buffer = ImageBuffer::create(FloatSize(canvasBase().width(), canvasBase().height()), RenderingMode::Unaccelerated, RenderingPurpose::Unspecified, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8);
    canvasBase().setImageBufferAndMarkDirty(WTFMove(buffer));
}

RefPtr<ImageBuffer> ImageBitmapRenderingContext::transferToImageBuffer()
{
    if (!canvasBase().hasCreatedImageBuffer())
        return canvasBase().allocateImageBuffer();
    RefPtr result = canvasBase().buffer();
    if (!result)
        return nullptr;
    setBlank();
    return result;
}

}
