/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

#include "GLDisplay.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

typedef intptr_t EGLAttrib;
typedef void *EGLClientBuffer;
typedef void *EGLContext;
typedef void *EGLDisplay;
typedef void *EGLImage;
typedef unsigned EGLenum;

#if ENABLE(VIDEO) && USE(GSTREAMER)
#include "GRefPtrGStreamer.h"

typedef struct _GstGLContext GstGLContext;
typedef struct _GstGLDisplay GstGLDisplay;
#endif // ENABLE(VIDEO) && USE(GSTREAMER)

#if USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/gpu/ganesh/GrDirectContext.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#include <wtf/ThreadSafeWeakHashSet.h>
#endif

namespace WebCore {

class GLContext;
#if USE(SKIA)
class SkiaGLContext;
#endif

class PlatformDisplay {
    WTF_MAKE_TZONE_ALLOCATED(PlatformDisplay);
    WTF_MAKE_NONCOPYABLE(PlatformDisplay);
public:
    WEBCORE_EXPORT static PlatformDisplay& sharedDisplay();
#if !PLATFORM(WIN)
    WEBCORE_EXPORT static void setSharedDisplay(std::unique_ptr<PlatformDisplay>&&);
    WEBCORE_EXPORT static PlatformDisplay* sharedDisplayIfExists();
#endif
    virtual ~PlatformDisplay();

    enum class Type {
#if PLATFORM(WIN)
        Windows,
#endif
#if USE(WPE_RENDERER)
        WPE,
#endif
        Surfaceless,
#if USE(GBM)
        GBM,
#endif
#if PLATFORM(GTK)
        Default,
#endif
    };

    virtual Type type() const = 0;

    WEBCORE_EXPORT GLContext* sharingGLContext();
    void clearSharingGLContext();
    EGLDisplay eglDisplay() const;
    bool eglCheckVersion(int major, int minor) const;

    const GLDisplay::Extensions& eglExtensions() const;

    EGLImage createEGLImage(EGLContext, EGLenum target, EGLClientBuffer, const Vector<EGLAttrib>&) const;
    bool destroyEGLImage(EGLImage) const;
#if USE(GBM)
    const Vector<GLDisplay::DMABufFormat>& dmabufFormats();
#if USE(GSTREAMER)
    const Vector<GLDisplay::DMABufFormat>& dmabufFormatsForVideo();
#endif
#endif

#if ENABLE(WEBGL)
    EGLDisplay angleEGLDisplay() const;
    EGLContext angleSharingGLContext();
#endif

#if ENABLE(VIDEO) && USE(GSTREAMER)
    GstGLDisplay* gstGLDisplay() const;
    GstGLContext* gstGLContext() const;
    void clearGStreamerGLState();
#endif

#if USE(SKIA)
    GLContext* skiaGLContext();
    GrDirectContext* skiaGrContext();
    unsigned msaaSampleCount() const;
#endif

protected:
    explicit PlatformDisplay(std::unique_ptr<GLDisplay>&&);

    std::unique_ptr<GLDisplay> m_eglDisplay;
    std::unique_ptr<GLContext> m_sharingGLContext;

#if ENABLE(WEBGL) && !PLATFORM(WIN)
    std::optional<int> m_anglePlatform;
    void* m_angleNativeDisplay { nullptr };
#endif

private:
#if USE(SKIA)
    void invalidateSkiaGLContexts();
#endif

#if ENABLE(WEBGL) && !PLATFORM(WIN)
    void clearANGLESharingGLContext();
#endif

    void terminateEGLDisplay();

#if ENABLE(WEBGL) && !PLATFORM(WIN)
    mutable EGLDisplay m_angleEGLDisplay { nullptr };
    EGLContext m_angleSharingGLContext { nullptr };
#endif

#if ENABLE(VIDEO) && USE(GSTREAMER)
    mutable GRefPtr<GstGLDisplay> m_gstGLDisplay;
    mutable GRefPtr<GstGLContext> m_gstGLContext;
#endif

#if USE(SKIA)
    ThreadSafeWeakHashSet<SkiaGLContext> m_skiaGLContexts;
#endif
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_PLATFORM_DISPLAY(ToClassName, DisplayType) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::PlatformDisplay& display) { return display.type() == WebCore::PlatformDisplay::Type::DisplayType; } \
SPECIALIZE_TYPE_TRAITS_END()
