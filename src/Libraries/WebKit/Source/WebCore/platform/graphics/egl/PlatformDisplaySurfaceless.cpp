/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#include "PlatformDisplaySurfaceless.h"

#include "GLContext.h"
#include <epoxy/egl.h>

namespace WebCore {

std::unique_ptr<PlatformDisplaySurfaceless> PlatformDisplaySurfaceless::create()
{
    const char* extensions = eglQueryString(nullptr, EGL_EXTENSIONS);
    if (!GLContext::isExtensionSupported(extensions, "EGL_MESA_platform_surfaceless"))
        return nullptr;

    std::unique_ptr<GLDisplay> glDisplay;
    if (GLContext::isExtensionSupported(extensions, "EGL_EXT_platform_base"))
        glDisplay = GLDisplay::create(eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr));
    else if (GLContext::isExtensionSupported(extensions, "EGL_KHR_platform_base"))
        glDisplay = GLDisplay::create(eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr));

    if (!glDisplay) {
        WTFLogAlways("Could not create surfaceless EGL display: %s. Aborting...", GLContext::lastErrorString());
        CRASH();
    }

    return std::unique_ptr<PlatformDisplaySurfaceless>(new PlatformDisplaySurfaceless(WTFMove(glDisplay)));
}

PlatformDisplaySurfaceless::PlatformDisplaySurfaceless(std::unique_ptr<GLDisplay>&& glDisplay)
    : PlatformDisplay(WTFMove(glDisplay))
{
#if ENABLE(WEBGL)
    m_anglePlatform = EGL_PLATFORM_SURFACELESS_MESA;
    m_angleNativeDisplay = EGL_DEFAULT_DISPLAY;
#endif
}

PlatformDisplaySurfaceless::~PlatformDisplaySurfaceless()
{
}

} // namespace WebCore
