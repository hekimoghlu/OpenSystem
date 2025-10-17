/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#include "CairoPaintingContext.h"

#if USE(CAIRO)
#include "CairoOperationRecorder.h"
#include "CoordinatedTileBuffer.h"
#include "GraphicsContext.h"
#include "GraphicsContextCairo.h"
#include <cairo.h>
#include <utility>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Cairo {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PaintingContext);

std::unique_ptr<PaintingContext> PaintingContext::createForPainting(WebCore::CoordinatedTileBuffer& buffer)
{
    return std::unique_ptr<PaintingContext>(new PaintingContext(buffer));
}

std::unique_ptr<PaintingContext> PaintingContext::createForRecording(PaintingOperations& paintingOperations)
{
    return std::unique_ptr<PaintingContext>(new PaintingContext(paintingOperations));
}

PaintingContext::PaintingContext(WebCore::CoordinatedTileBuffer& baseBuffer)
{
    // All buffers used for painting with Cairo are unaccelerated.
    auto& buffer = static_cast<WebCore::CoordinatedUnacceleratedTileBuffer&>(baseBuffer);

    // Balanced by the deref in the s_bufferKey user data destroy callback.
    buffer.ref();

    m_surface = adoptRef(cairo_image_surface_create_for_data(buffer.data(),
        CAIRO_FORMAT_ARGB32, baseBuffer.size().width(), baseBuffer.size().height(), buffer.stride()));

    static cairo_user_data_key_t s_bufferKey;
    cairo_surface_set_user_data(m_surface.get(), &s_bufferKey,
        new std::pair<WebCore::CoordinatedTileBuffer*, PaintingContext*> { &buffer, this }, [](void* data) {
            auto* userData = static_cast<std::pair<WebCore::CoordinatedTileBuffer*, PaintingContext*>*>(data);

            // Deref the CoordinatedTileBuffer object.
            userData->first->deref();
#if ASSERT_ENABLED
            // Mark the deletion of the cairo_surface_t object associated with this
            // PaintingContextCairo as complete. This way we check that the cairo_surface_t
            // object doesn't outlive the PaintingContextCairo through which it was used.
            userData->second->m_deletionComplete = true;
#endif
            delete userData;
        });

    m_graphicsContext = makeUnique<WebCore::GraphicsContextCairo>(m_surface.get());
}

PaintingContext::PaintingContext(PaintingOperations& paintingOperations)
    : m_graphicsContext(makeUnique<OperationRecorder>(paintingOperations))
{
}

PaintingContext::~PaintingContext()
{
    if (!m_surface)
        return;

    cairo_surface_flush(m_surface.get());

    m_graphicsContext = nullptr;
    m_surface = nullptr;

    // With all the Cairo references purged, the cairo_surface_t object should be destroyed
    // as well. This is checked by asserting that m_deletionComplete is true, which should
    // be the case if the s_bufferKey user data destroy callback has been invoked upon the
    // cairo_surface_t destruction.
    ASSERT(m_deletionComplete);
}

void PaintingContext::replay(const PaintingOperations& paintingOperations)
{
    ASSERT(m_surface);
    auto& context = *m_graphicsContext->platformContext();
    for (auto& operation : paintingOperations)
        operation->execute(context);
}

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
