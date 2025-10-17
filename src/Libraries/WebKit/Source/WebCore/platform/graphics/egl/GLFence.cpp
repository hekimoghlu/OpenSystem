/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#include "GLFence.h"

#include "GLContext.h"
#include "GLFenceEGL.h"
#include "GLFenceGL.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(LIBEPOXY)
#include <epoxy/gl.h>
#else
#include <GLES2/gl2.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GLFence);

const GLFence::Capabilities& GLFence::capabilities()
{
    static Capabilities capabilities;
#if HAVE(GL_FENCE)
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        auto& display = PlatformDisplay::sharedDisplay();
        const auto& extensions = display.eglExtensions();
        if (display.eglCheckVersion(1, 5)) {
            capabilities.eglSupported = true;
            capabilities.eglServerWaitSupported = true;
        } else {
            capabilities.eglSupported = extensions.KHR_fence_sync;
            capabilities.eglServerWaitSupported = extensions.KHR_wait_sync;
        }
#if OS(UNIX)
        capabilities.eglExportableSupported = extensions.ANDROID_native_fence_sync;
#endif
        capabilities.glSupported = GLContext::versionFromString(reinterpret_cast<const char*>(glGetString(GL_VERSION))) >= 300;
    });
#endif
    return capabilities;
}

bool GLFence::isSupported()
{
    const auto& fenceCapabilities = capabilities();
    return fenceCapabilities.eglSupported || fenceCapabilities.glSupported;
}

std::unique_ptr<GLFence> GLFence::create()
{
#if HAVE(GL_FENCE)
    if (!GLContextWrapper::currentContext())
        return nullptr;

    const auto& fenceCapabilities = capabilities();
    if (fenceCapabilities.eglSupported && fenceCapabilities.eglServerWaitSupported)
        return GLFenceEGL::create();

    if (fenceCapabilities.glSupported)
        return GLFenceGL::create();

    if (fenceCapabilities.eglSupported)
        return GLFenceEGL::create();
#endif
    return nullptr;
}

#if OS(UNIX)
std::unique_ptr<GLFence> GLFence::createExportable()
{
#if HAVE(GL_FENCE)
    if (!GLContextWrapper::currentContext())
        return nullptr;

    const auto& fenceCapabilities = capabilities();
    if (fenceCapabilities.eglSupported && fenceCapabilities.eglExportableSupported)
        return GLFenceEGL::createExportable();
#endif
    return nullptr;
}

std::unique_ptr<GLFence> GLFence::importFD(UnixFileDescriptor&& fd)
{
#if HAVE(GL_FENCE)
    if (!GLContextWrapper::currentContext())
        return nullptr;

    const auto& fenceCapabilities = capabilities();
    if (fenceCapabilities.eglSupported && fenceCapabilities.eglExportableSupported)
        return GLFenceEGL::importFD(WTFMove(fd));
#else
    UNUSED_PARAM(fd);
#endif
    return nullptr;
}
#endif

} // namespace WebCore
