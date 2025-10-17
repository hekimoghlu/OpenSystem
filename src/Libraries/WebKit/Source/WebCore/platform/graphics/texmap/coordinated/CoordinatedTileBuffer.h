/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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

#if USE(COORDINATED_GRAPHICS)

#include "IntSize.h"
#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/MallocSpan.h>
#include <wtf/Ref.h>
#include <wtf/ThreadSafeRefCounted.h>

#if USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkCanvas.h>
#include <skia/core/SkSurface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif

namespace WebCore {
class BitmapTexture;
class GLFence;
enum class PixelFormat : uint8_t;

class CoordinatedTileBuffer : public ThreadSafeRefCounted<CoordinatedTileBuffer> {
public:
    enum Flag {
        NoFlags = 0,
        SupportsAlpha = 1 << 0,
    };
    using Flags = unsigned;

    WEBCORE_EXPORT virtual ~CoordinatedTileBuffer();

    virtual IntSize size() const = 0;
    virtual bool isBackedByOpenGL() const = 0;

    bool supportsAlpha() const { return m_flags & SupportsAlpha; }

    virtual void beginPainting();
    virtual void completePainting();
    void waitUntilPaintingComplete();

#if USE(SKIA)
    SkCanvas* canvas();
#endif

    static void resetMemoryUsage();
    static double getMemoryUsage();

protected:
    explicit CoordinatedTileBuffer(Flags);

#if USE(SKIA)
    virtual bool tryEnsureSurface() = 0;
    sk_sp<SkSurface> m_surface;
#endif

    static Lock s_layersMemoryUsageLock;
    static double s_currentLayersMemoryUsage;
    static double s_maxLayersMemoryUsage;

    enum class PaintingState {
        InProgress,
        Complete
    };

    struct {
        Lock lock;
        Condition condition;
        PaintingState state { PaintingState::Complete };
    } m_painting;

private:
    Flags m_flags;
};

class CoordinatedUnacceleratedTileBuffer final : public CoordinatedTileBuffer {
public:
    WEBCORE_EXPORT static Ref<CoordinatedTileBuffer> create(const IntSize&, Flags);
    WEBCORE_EXPORT virtual ~CoordinatedUnacceleratedTileBuffer();

    int stride() const { return m_size.width() * 4; }

    const unsigned char* data() const { return m_data.span().data(); }
    unsigned char* data() { return m_data.mutableSpan().data(); }

    PixelFormat pixelFormat() const;

private:
    CoordinatedUnacceleratedTileBuffer(const IntSize&, Flags);

    bool isBackedByOpenGL() const final { return false; }
    IntSize size() const final { return m_size; }

#if USE(SKIA)
    bool tryEnsureSurface() final;
#endif

    MallocSpan<unsigned char> m_data;
    IntSize m_size;

    enum class PaintingState {
        InProgress,
        Complete
    };

    struct {
        Lock lock;
        Condition condition;
        PaintingState state { PaintingState::Complete };
    } m_painting;
};

#if USE(SKIA)
class CoordinatedAcceleratedTileBuffer final : public CoordinatedTileBuffer {
public:
    WEBCORE_EXPORT static Ref<CoordinatedTileBuffer> create(Ref<BitmapTexture>&&);
    WEBCORE_EXPORT virtual ~CoordinatedAcceleratedTileBuffer();

    BitmapTexture& texture() const { return m_texture.get(); }
    void serverWait();

private:
    CoordinatedAcceleratedTileBuffer(Ref<BitmapTexture>&&, Flags);

    bool isBackedByOpenGL() const final { return true; }
    IntSize size() const final;

    bool tryEnsureSurface() final;
    void completePainting() final;

    Ref<BitmapTexture> m_texture;
    std::unique_ptr<GLFence> m_fence;
};
#endif

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
