/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#include "PlatformDisplayGBM.h"

#if USE(GBM)
#include "GLContext.h"
#include <epoxy/egl.h>
#include <gbm.h>

namespace WebCore {

std::unique_ptr<PlatformDisplayGBM> PlatformDisplayGBM::create(struct gbm_device* device)
{
    const char* extensions = eglQueryString(nullptr, EGL_EXTENSIONS);
    if (!GLContext::isExtensionSupported(extensions, "EGL_KHR_platform_gbm"))
        return nullptr;

    std::unique_ptr<GLDisplay> glDisplay;
    if (GLContext::isExtensionSupported(extensions, "EGL_EXT_platform_base"))
        glDisplay = GLDisplay::create(eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR, device, nullptr));
    else if (GLContext::isExtensionSupported(extensions, "EGL_KHR_platform_base"))
        glDisplay = GLDisplay::create(eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, device, nullptr));

    if (!glDisplay) {
        WTFLogAlways("Could not create GBM EGL display: %s. Aborting...", GLContext::lastErrorString());
        CRASH();
    }

    return std::unique_ptr<PlatformDisplayGBM>(new PlatformDisplayGBM(WTFMove(glDisplay), device));
}

PlatformDisplayGBM::PlatformDisplayGBM(std::unique_ptr<GLDisplay>&& glDisplay, struct gbm_device* device)
    : PlatformDisplay(WTFMove(glDisplay))
{
#if ENABLE(WEBGL)
    m_anglePlatform = EGL_PLATFORM_GBM_KHR;
    m_angleNativeDisplay = device;
#endif
}

PlatformDisplayGBM::~PlatformDisplayGBM()
{
}

} // namespace WebCore

#endif // USE(GBM)
