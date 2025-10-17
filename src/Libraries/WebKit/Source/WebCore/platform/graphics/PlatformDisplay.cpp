/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#include "PlatformDisplay.h"

#include "GLContext.h"
#include <cstdlib>
#include <mutex>
#include <wtf/HashSet.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(WIN)
#include "PlatformDisplayWin.h"
#endif

#if USE(LIBEPOXY)
#include <epoxy/egl.h>
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PlatformDisplay);

#if PLATFORM(WIN)
PlatformDisplay& PlatformDisplay::sharedDisplay()
{
    // ANGLE D3D renderer isn't thread-safe. Don't destruct it on non-main threads which calls _exit().
    static PlatformDisplay* display = PlatformDisplayWin::create().release();
    return *display;
}
#else
IGNORE_CLANG_WARNINGS_BEGIN("exit-time-destructors")
static std::unique_ptr<PlatformDisplay> s_sharedDisplay;
IGNORE_CLANG_WARNINGS_END

void PlatformDisplay::setSharedDisplay(std::unique_ptr<PlatformDisplay>&& display)
{
    RELEASE_ASSERT(!s_sharedDisplay);
    s_sharedDisplay = WTFMove(display);
}

PlatformDisplay& PlatformDisplay::sharedDisplay()
{
    RELEASE_ASSERT(s_sharedDisplay);
    return *s_sharedDisplay;
}

PlatformDisplay* PlatformDisplay::sharedDisplayIfExists()
{
    return s_sharedDisplay.get();
}
#endif

static UncheckedKeyHashSet<PlatformDisplay*>& eglDisplays()
{
    static NeverDestroyed<UncheckedKeyHashSet<PlatformDisplay*>> displays;
    return displays;
}

PlatformDisplay::PlatformDisplay(std::unique_ptr<GLDisplay>&& glDisplay)
    : m_eglDisplay(WTFMove(glDisplay))
{
    RELEASE_ASSERT(m_eglDisplay);

    eglDisplays().add(this);

#if !PLATFORM(WIN)
    static bool eglAtexitHandlerInitialized = false;
    if (!eglAtexitHandlerInitialized) {
        // EGL registers atexit handlers to cleanup its global display list.
        // Since the global PlatformDisplay instance is created before,
        // when the PlatformDisplay destructor is called, EGL has already removed the
        // display from the list, causing eglTerminate() to crash. So, here we register
        // our own atexit handler, after EGL has been initialized and after the global
        // instance has been created to ensure we call eglTerminate() before the other
        // EGL atexit handlers and the PlatformDisplay destructor.
        // See https://bugs.webkit.org/show_bug.cgi?id=157973.
        eglAtexitHandlerInitialized = true;
        std::atexit([] {
            while (!eglDisplays().isEmpty()) {
                auto* display = eglDisplays().takeAny();
                display->terminateEGLDisplay();
            }
        });
    }
#endif
}

PlatformDisplay::~PlatformDisplay()
{
    if (eglDisplays().remove(this))
        m_eglDisplay->terminate();
}

GLContext* PlatformDisplay::sharingGLContext()
{
    if (!m_sharingGLContext)
        m_sharingGLContext = GLContext::createSharing(*this);
    return m_sharingGLContext.get();
}

void PlatformDisplay::clearSharingGLContext()
{
#if USE(SKIA)
    invalidateSkiaGLContexts();
#endif
#if ENABLE(VIDEO) && USE(GSTREAMER)
    m_gstGLContext = nullptr;
#endif
#if ENABLE(WEBGL) && !PLATFORM(WIN)
    clearANGLESharingGLContext();
#endif
    m_sharingGLContext = nullptr;
}

EGLDisplay PlatformDisplay::eglDisplay() const
{
    return m_eglDisplay->eglDisplay();
}

bool PlatformDisplay::eglCheckVersion(int major, int minor) const
{
    return m_eglDisplay->checkVersion(major, minor);
}

const GLDisplay::Extensions& PlatformDisplay::eglExtensions() const
{
    return m_eglDisplay->extensions();
}

void PlatformDisplay::terminateEGLDisplay()
{
#if ENABLE(VIDEO) && USE(GSTREAMER)
    m_gstGLDisplay = nullptr;
#endif
    clearSharingGLContext();

    m_eglDisplay->terminate();
}

EGLImage PlatformDisplay::createEGLImage(EGLContext context, EGLenum target, EGLClientBuffer clientBuffer, const Vector<EGLAttrib>& attributes) const
{
    return m_eglDisplay->createImage(context, target, clientBuffer, attributes);
}

bool PlatformDisplay::destroyEGLImage(EGLImage image) const
{
    return m_eglDisplay->destroyImage(image);
}

#if USE(GBM)
const Vector<GLDisplay::DMABufFormat>& PlatformDisplay::dmabufFormats()
{
    return m_eglDisplay->dmabufFormats();
}

#if USE(GSTREAMER)
const Vector<GLDisplay::DMABufFormat>& PlatformDisplay::dmabufFormatsForVideo()
{
    return m_eglDisplay->dmabufFormatsForVideo();
}
#endif
#endif // USE(GBM)

} // namespace WebCore
