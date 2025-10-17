/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "WebDragClient.h"

#if ENABLE(DRAG_SUPPORT)

#include "ArgumentCodersGtk.h"
#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/DataTransfer.h>
#include <WebCore/DragData.h>
#include <WebCore/GraphicsContext.h>
#include <WebCore/Pasteboard.h>
#include <WebCore/SelectionData.h>
#include <WebCore/ShareableBitmap.h>

#if USE(CAIRO)
#include <WebCore/CairoOperations.h>
#include <cairo.h>
#endif

namespace WebKit {
using namespace WebCore;

#if USE(CAIRO)
static RefPtr<ShareableBitmap> convertCairoSurfaceToShareableBitmap(cairo_surface_t* surface)
{
    if (!surface)
        return nullptr;

    IntSize imageSize(cairo_image_surface_get_width(surface), cairo_image_surface_get_height(surface));
    auto bitmap = ShareableBitmap::create({ imageSize });
    auto graphicsContext = bitmap->createGraphicsContext();

    ASSERT(graphicsContext->hasPlatformContext());
    auto& state = graphicsContext->state();
    Cairo::drawSurface(*graphicsContext->platformContext(), surface, IntRect(IntPoint(), imageSize), IntRect(IntPoint(), imageSize), state.imageInterpolationQuality(), state.alpha(), Cairo::ShadowState(state));
    return bitmap;
}
#endif

#if USE(SKIA)
static RefPtr<ShareableBitmap> convertSkiaImageToShareableBitmap(SkImage* image)
{
    if (!image)
        return nullptr;

    IntSize imageSize(image->width(), image->height());
    RefPtr bitmap = ShareableBitmap::create({ imageSize });
    auto graphicsContext = bitmap->createGraphicsContext();

    ASSERT(graphicsContext->hasPlatformContext());
    graphicsContext->platformContext()->drawImage(image, 0, 0);

    return bitmap;
}
#endif

void WebDragClient::didConcludeEditDrag()
{
}

void WebDragClient::startDrag(DragItem dragItem, DataTransfer& dataTransfer, Frame&)
{
    std::optional<ShareableBitmap::Handle> handle;
    auto* dragSurface = dragItem.image.get().get();
    RefPtr<ShareableBitmap> bitmap;

#if USE(CAIRO)
    bitmap = convertCairoSurfaceToShareableBitmap(dragSurface);
#elif USE(SKIA)
    bitmap = convertSkiaImageToShareableBitmap(dragSurface);
#endif

    if (bitmap) {
        handle = bitmap->createHandle();

        // If we have a bitmap, but cannot create a handle to it, we fail early.
        if (!handle)
            return;
    }

    m_page->willStartDrag();
    m_page->send(Messages::WebPageProxy::StartDrag(dataTransfer.pasteboard().selectionData(), dataTransfer.sourceOperationMask(), WTFMove(handle), dataTransfer.dragLocation()));
}

}; // namespace WebKit.

#endif // ENABLE(DRAG_SUPPORT)
