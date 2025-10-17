/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

#if HAVE(IOSURFACE)

#include "IOSurface.h"
#include "NativeImage.h"

namespace WebCore {

// Move-only value type holding a IOSurface that will be used in by drawing to it
// as well as reading from it via CG.
// Important subtle expected behavior is to migrate the existing CGImages from
// IOSurfaces to main memory when the drawing buffer instance is destroyed. This
// is to prevent long-lived images reserving IOSurfaces.
class IOSurfaceDrawingBuffer {
public:
    IOSurfaceDrawingBuffer() = default;
    IOSurfaceDrawingBuffer(IOSurfaceDrawingBuffer&&);
    explicit IOSurfaceDrawingBuffer(std::unique_ptr<IOSurface>&&);
    IOSurfaceDrawingBuffer& operator=(IOSurfaceDrawingBuffer&&);
    operator bool() const { return !!m_surface; }
    IOSurface* surface() const { return m_surface.get(); }

    IntSize size() const;

    // Returns true if surface cannot be modified because it's in
    // cross-process use, and copy-on-write would not work.
    bool isInUse() const;

    // Should be called always when writing to the surface.
    void prepareForWrite();

    // Creates a copy of current contents.
    RefPtr<NativeImage> copyNativeImage() const;
private:
    void forceCopy();
    std::unique_ptr<IOSurface> m_surface;
    mutable RetainPtr<CGContextRef> m_copyOnWriteContext;
    mutable bool m_needCopy { false };
};

inline IOSurfaceDrawingBuffer::IOSurfaceDrawingBuffer(IOSurfaceDrawingBuffer&& other)
    : m_surface(WTFMove(other.m_surface))
    , m_copyOnWriteContext(WTFMove(other.m_copyOnWriteContext))
    , m_needCopy(std::exchange(other.m_needCopy, false))
{
}

inline IOSurfaceDrawingBuffer::IOSurfaceDrawingBuffer(std::unique_ptr<IOSurface>&& surface)
    : m_surface(WTFMove(surface))
{
}

inline IOSurfaceDrawingBuffer& IOSurfaceDrawingBuffer::operator=(IOSurfaceDrawingBuffer&& other)
{
    m_surface = WTFMove(other.m_surface);
    m_copyOnWriteContext = WTFMove(other.m_copyOnWriteContext);
    m_needCopy = std::exchange(other.m_needCopy, false);
    return *this;
}

inline void IOSurfaceDrawingBuffer::prepareForWrite()
{
    if (m_needCopy)
        forceCopy();
}

inline bool IOSurfaceDrawingBuffer::isInUse() const
{
    if (!m_surface)
        return false;
    return m_surface->isInUse();
}

inline IntSize IOSurfaceDrawingBuffer::size() const
{
    if (!m_surface)
        return { };
    return m_surface->size();
}

}

#endif
