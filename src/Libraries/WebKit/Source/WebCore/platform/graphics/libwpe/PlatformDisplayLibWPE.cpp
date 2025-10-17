/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#include "PlatformDisplayLibWPE.h"

#if USE(WPE_RENDERER)

#include "GLContext.h"

#if USE(LIBEPOXY)
// FIXME: For now default to the GBM EGL platform, but this should really be
// somehow deducible from the build configuration.
#define __GBM__ 1
#include <epoxy/egl.h>
#else
#if PLATFORM(WAYLAND)
// These includes need to be in this order because wayland-egl.h defines WL_EGL_PLATFORM
// and eglplatform.h, included by egl.h, checks that to decide whether it's Wayland platform.
#include <wayland-egl.h>
#endif
#include <EGL/egl.h>
#endif

#include <wpe/wpe-egl.h>

#ifndef EGL_EXT_platform_base
#define EGL_EXT_platform_base 1
typedef EGLDisplay (EGLAPIENTRYP PFNEGLGETPLATFORMDISPLAYEXTPROC) (EGLenum platform, void *native_display, const EGLint *attrib_list);
#endif

namespace WebCore {

std::unique_ptr<PlatformDisplayLibWPE> PlatformDisplayLibWPE::create(int hostFd)
{
    auto* backend = wpe_renderer_backend_egl_create(hostFd);
    if (!backend)
        return nullptr;

    EGLNativeDisplayType eglNativeDisplay = wpe_renderer_backend_egl_get_native_display(backend);

    std::unique_ptr<GLDisplay> glDisplay;
#if WPE_CHECK_VERSION(1, 1, 0)
    uint32_t eglPlatform = wpe_renderer_backend_egl_get_platform(backend);
    if (eglPlatform) {
        using GetPlatformDisplayType = PFNEGLGETPLATFORMDISPLAYEXTPROC;
        GetPlatformDisplayType getPlatformDisplay =
            [] {
                const char* extensions = eglQueryString(nullptr, EGL_EXTENSIONS);
                if (GLContext::isExtensionSupported(extensions, "EGL_EXT_platform_base")) {
                    if (auto extension = reinterpret_cast<GetPlatformDisplayType>(eglGetProcAddress("eglGetPlatformDisplayEXT")))
                        return extension;
                }
                if (GLContext::isExtensionSupported(extensions, "EGL_KHR_platform_base")) {
                    if (auto extension = reinterpret_cast<GetPlatformDisplayType>(eglGetProcAddress("eglGetPlatformDisplay")))
                        return extension;
                }
                return GetPlatformDisplayType(nullptr);
            }();

        if (getPlatformDisplay)
            glDisplay = GLDisplay::create(getPlatformDisplay(eglPlatform, eglNativeDisplay, nullptr));
    }
#endif

    if (!glDisplay)
        glDisplay = GLDisplay::create(eglGetDisplay(eglNativeDisplay));

    if (!glDisplay) {
        WTFLogAlways("Could not create WPE EGL display: %s. Aborting...", GLContext::lastErrorString());
        CRASH();
    }

    return std::unique_ptr<PlatformDisplayLibWPE>(new PlatformDisplayLibWPE(WTFMove(glDisplay), backend));
}

PlatformDisplayLibWPE::PlatformDisplayLibWPE(std::unique_ptr<GLDisplay>&& glDisplay, struct wpe_renderer_backend_egl* backend)
    : PlatformDisplay(WTFMove(glDisplay))
    , m_backend(backend)
{
#if ENABLE(WEBGL)
#if WPE_CHECK_VERSION(1, 1, 0)
    m_anglePlatform = wpe_renderer_backend_egl_get_platform(m_backend);
#endif
    m_angleNativeDisplay = wpe_renderer_backend_egl_get_native_display(m_backend);
#endif
}

PlatformDisplayLibWPE::~PlatformDisplayLibWPE()
{
    if (m_backend)
        wpe_renderer_backend_egl_destroy(m_backend);
}

} // namespace WebCore

#endif // USE(WPE_RENDERER)
