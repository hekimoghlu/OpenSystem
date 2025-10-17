/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "GLContext.h"

#if USE(WPE_RENDERER)
#include "PlatformDisplayLibWPE.h"

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

namespace WebCore {

GLContext::GLContext(PlatformDisplay& display, EGLContext context, EGLSurface surface, EGLConfig config, struct wpe_renderer_backend_egl_offscreen_target* target)
    : m_display(display)
    , m_context(context)
    , m_surface(surface)
    , m_config(config)
    , m_type(WindowSurface)
    , m_wpeTarget(target)
{
}

EGLSurface GLContext::createWindowSurfaceWPE(EGLDisplay display, EGLConfig config, GLNativeWindowType window)
{
    // EGLNativeWindowType changes depending on the EGL implementation, reinterpret_cast
    // would work for pointers, and static_cast for numeric types only; so use a plain
    // C cast expression which works in all possible cases.
    return eglCreateWindowSurface(display, config, (EGLNativeWindowType) window, nullptr);
}

std::unique_ptr<GLContext> GLContext::createWPEContext(PlatformDisplay& platformDisplay, EGLContext sharingContext)
{
    EGLConfig config;
    if (!getEGLConfig(platformDisplay, &config, WindowSurface)) {
        WTFLogAlways("Cannot obtain EGL WPE context configuration: %s\n", lastErrorString());
        return nullptr;
    }

    auto* target = wpe_renderer_backend_egl_offscreen_target_create();
    if (!target)
        return nullptr;

    wpe_renderer_backend_egl_offscreen_target_initialize(target, downcast<PlatformDisplayLibWPE>(platformDisplay).backend());
    EGLNativeWindowType window = wpe_renderer_backend_egl_offscreen_target_get_native_window(target);
    if (!window) {
        wpe_renderer_backend_egl_offscreen_target_destroy(target);
        return nullptr;
    }

    EGLContext context = createContextForEGLVersion(platformDisplay, config, sharingContext);
    if (context == EGL_NO_CONTEXT) {
        WTFLogAlways("Cannot create EGL WPE context: %s\n", lastErrorString());
        wpe_renderer_backend_egl_offscreen_target_destroy(target);
        return nullptr;
    }

    EGLDisplay display = platformDisplay.eglDisplay();
    EGLSurface surface = eglCreateWindowSurface(display, config, static_cast<EGLNativeWindowType>(window), nullptr);
    if (surface == EGL_NO_SURFACE) {
        WTFLogAlways("Cannot create EGL WPE window surface: %s\n", lastErrorString());
        eglDestroyContext(display, context);
        wpe_renderer_backend_egl_offscreen_target_destroy(target);
        return nullptr;
    }

    return makeUnique<GLContext>(platformDisplay, context, surface, config, target);
}

void GLContext::destroyWPETarget()
{
    if (m_wpeTarget)
        wpe_renderer_backend_egl_offscreen_target_destroy(m_wpeTarget);
}

} // namespace WebCore

#endif // USE(WPE_RENDERER)
