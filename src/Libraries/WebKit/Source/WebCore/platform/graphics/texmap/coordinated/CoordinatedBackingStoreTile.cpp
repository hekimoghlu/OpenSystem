/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#include "CoordinatedBackingStoreTile.h"

#if USE(COORDINATED_GRAPHICS)
#include "BitmapTexture.h"
#include "CoordinatedTileBuffer.h"
#include "GraphicsLayer.h"
#include "TextureMapper.h"
#include <wtf/SystemTracing.h>

namespace WebCore {

CoordinatedBackingStoreTile::CoordinatedBackingStoreTile(float scale)
    : m_scale(scale)
{
}

CoordinatedBackingStoreTile::~CoordinatedBackingStoreTile() = default;

void CoordinatedBackingStoreTile::addUpdate(Update&& update)
{
    m_updates.append(WTFMove(update));
}

void CoordinatedBackingStoreTile::processPendingUpdates(TextureMapper& textureMapper)
{
    auto updates = WTFMove(m_updates);
    auto updatesCount = updates.size();
    if (!updatesCount)
        return;

    WTFBeginSignpost(this, CoordinatedSwapBuffers, "%lu updates", updatesCount);
    for (unsigned updateIndex = 0; updateIndex < updatesCount; ++updateIndex) {
        auto& update = updates[updateIndex];
        if (!update.buffer)
            continue;

        WTFBeginSignpost(this, CoordinatedSwapBuffer, "%u/%lu, rect %ix%i+%i+%i", updateIndex + 1, updatesCount, update.tileRect.x(), update.tileRect.y(), update.tileRect.width(), update.tileRect.height());

        ASSERT(textureMapper.maxTextureSize().width() >= update.tileRect.size().width());
        ASSERT(textureMapper.maxTextureSize().height() >= update.tileRect.size().height());

        FloatRect unscaledTileRect(update.tileRect);
        unscaledTileRect.scale(1. / m_scale);

        OptionSet<BitmapTexture::Flags> flags;
        if (update.buffer->supportsAlpha())
            flags.add(BitmapTexture::Flags::SupportsAlpha);

        WTFBeginSignpost(this, AcquireTexture);
        if (!m_texture || unscaledTileRect != m_rect) {
            m_rect = unscaledTileRect;
            m_texture = textureMapper.acquireTextureFromPool(update.tileRect.size(), flags);
        } else if (update.buffer->supportsAlpha() == m_texture->isOpaque())
            m_texture->reset(update.tileRect.size(), flags);
        WTFEndSignpost(this, AcquireTexture);

        WTFBeginSignpost(this, WaitPaintingCompletion);
        update.buffer->waitUntilPaintingComplete();
        WTFEndSignpost(this, WaitPaintingCompletion);

#if USE(SKIA)
        if (update.buffer->isBackedByOpenGL()) {
            WTFBeginSignpost(this, CopyTextureGPUToGPU);
            auto& buffer = static_cast<CoordinatedAcceleratedTileBuffer&>(*update.buffer);
            buffer.serverWait();

            // Fast path: whole tile content changed -- take ownership of the incoming texture, replacing the existing tile buffer (avoiding texture copies).
            if (update.sourceRect.size() == update.tileRect.size()) {
                ASSERT(update.sourceRect.location().isZero());
                m_texture->swapTexture(buffer.texture());
            } else
                m_texture->copyFromExternalTexture(buffer.texture().id(), update.sourceRect, toIntSize(update.bufferOffset));

            update.buffer = nullptr;
            WTFEndSignpost(this, CopyTextureGPUToGPU);
            WTFEndSignpost(this, CoordinatedSwapBuffer);
            continue;
        }
#endif

        WTFBeginSignpost(this, CopyTextureCPUToGPU);
        ASSERT(!update.buffer->isBackedByOpenGL());
        auto& buffer = static_cast<CoordinatedUnacceleratedTileBuffer&>(*update.buffer);
        m_texture->updateContents(buffer.data(), update.sourceRect, update.bufferOffset, buffer.stride(), buffer.pixelFormat());
        update.buffer = nullptr;
        WTFEndSignpost(this, CopyTextureCPUToGPU);

        WTFEndSignpost(this, CoordinatedSwapBuffer);
    }
    WTFEndSignpost(this, CoordinatedSwapBuffers);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
