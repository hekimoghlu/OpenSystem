/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
#include "PlatformDisplayWin.h"

#include "GLContext.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>

namespace WebCore {

std::unique_ptr<PlatformDisplayWin> PlatformDisplayWin::create()
{
    EGLint attributes[] = {
        // Disable debug layers that makes some tests fail because
        // ANGLE fills uninitialized buffers with
        // kDebugColorInitClearValue in debug builds.
        EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED_ANGLE,
        EGL_FALSE,
        EGL_NONE,
    };
    auto glDisplay = GLDisplay::create(eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, EGL_DEFAULT_DISPLAY, attributes));
    if (!glDisplay) {
        WTFLogAlways("Could not create EGL display: %s. Aborting...", GLContext::lastErrorString());
        CRASH();
    }

    return std::unique_ptr<PlatformDisplayWin>(new PlatformDisplayWin(WTFMove(glDisplay)));
}

PlatformDisplayWin::PlatformDisplayWin(std::unique_ptr<GLDisplay>&& glDisplay)
    : PlatformDisplay(WTFMove(glDisplay))
{
}

} // namespace WebCore
