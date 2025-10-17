/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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

#include <optional>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

typedef intptr_t EGLAttrib;
typedef void* EGLClientBuffer;
typedef void* EGLContext;
typedef void* EGLDisplay;
typedef void* EGLImage;
typedef unsigned EGLenum;

namespace WebCore {

class GLDisplay {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(GLDisplay);
    WTF_MAKE_NONCOPYABLE(GLDisplay);
public:
    static std::unique_ptr<GLDisplay> create(EGLDisplay);
    explicit GLDisplay(EGLDisplay);
    ~GLDisplay() = default;

    EGLDisplay eglDisplay() const { return m_display; }
    bool checkVersion(int major, int minor) const;

    void terminate();

    EGLImage createImage(EGLContext, EGLenum, EGLClientBuffer, const Vector<EGLAttrib>&) const;
    bool destroyImage(EGLImage) const;

    struct Extensions {
        bool KHR_image_base { false };
        bool KHR_fence_sync { false };
        bool KHR_surfaceless_context { false };
        bool KHR_wait_sync { false };
        bool EXT_image_dma_buf_import { false };
        bool EXT_image_dma_buf_import_modifiers { false };
        bool MESA_image_dma_buf_export { false };
        bool ANDROID_native_fence_sync { false };
    };
    const Extensions& extensions() const { return m_extensions; }

#if USE(GBM)
    struct DMABufFormat {
        uint32_t fourcc { 0 };
        Vector<uint64_t, 1> modifiers;
    };
    const Vector<DMABufFormat>& dmabufFormats();
#if USE(GSTREAMER)
    const Vector<DMABufFormat>& dmabufFormatsForVideo();
#endif
#endif

private:
    EGLDisplay m_display { nullptr };
    struct {
        int major { 0 };
        int minor { 0 };
    } m_version;
    Extensions m_extensions;

#if USE(GBM)
    Vector<DMABufFormat> m_dmabufFormats;
#if USE(GSTREAMER)
    Vector<DMABufFormat> m_dmabufFormatsForVideo;
#endif
#endif
};

} // namespace WebCore
