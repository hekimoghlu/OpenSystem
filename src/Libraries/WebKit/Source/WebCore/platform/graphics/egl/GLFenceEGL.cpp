/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#include "GLFenceEGL.h"

#if HAVE(GL_FENCE)

#include "PlatformDisplay.h"
#include <wtf/Vector.h>

#if USE(LIBEPOXY)
#include <epoxy/egl.h>
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#endif

namespace WebCore {

static std::unique_ptr<GLFence> createEGLFence(EGLenum type, const Vector<EGLAttrib>& attributes)
{
    EGLSync sync = nullptr;
    auto& display = PlatformDisplay::sharedDisplay();
    if (display.eglCheckVersion(1, 5))
        sync = eglCreateSync(display.eglDisplay(), type, attributes.isEmpty() ? nullptr : attributes.data());
    else {
        Vector<EGLint> intAttributes = attributes.map<Vector<EGLint>>([] (EGLAttrib value) {
            return value;
        });
        sync = eglCreateSyncKHR(display.eglDisplay(), type, intAttributes.isEmpty() ? nullptr : intAttributes.data());
    }
    if (sync == EGL_NO_SYNC)
        return nullptr;

    glFlush();

#if OS(UNIX)
    bool isExportable = type == EGL_SYNC_NATIVE_FENCE_ANDROID;
#else
    bool isExportable = false;
#endif
    return makeUnique<GLFenceEGL>(sync, isExportable);
}

std::unique_ptr<GLFence> GLFenceEGL::create()
{
    return createEGLFence(EGL_SYNC_FENCE_KHR, { });
}

#if OS(UNIX)
std::unique_ptr<GLFence> GLFenceEGL::createExportable()
{
    return createEGLFence(EGL_SYNC_NATIVE_FENCE_ANDROID, { });
}

std::unique_ptr<GLFence> GLFenceEGL::importFD(UnixFileDescriptor&& fd)
{
    Vector<EGLAttrib> attributes = {
        EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fd.release(),
        EGL_NONE
    };
    return createEGLFence(EGL_SYNC_NATIVE_FENCE_ANDROID, attributes);
}
#endif

GLFenceEGL::GLFenceEGL(EGLSyncKHR sync, bool isExportable)
    : m_sync(sync)
    , m_isExportable(isExportable)
{
}

GLFenceEGL::~GLFenceEGL()
{
    auto& display = PlatformDisplay::sharedDisplay();
    if (display.eglCheckVersion(1, 5))
        eglDestroySync(display.eglDisplay(), m_sync);
    else
        eglDestroySyncKHR(display.eglDisplay(), m_sync);
}

void GLFenceEGL::clientWait()
{
    auto& display = PlatformDisplay::sharedDisplay();
    if (display.eglCheckVersion(1, 5))
        eglClientWaitSync(display.eglDisplay(), m_sync, 0, EGL_FOREVER);
    else
        eglClientWaitSyncKHR(display.eglDisplay(), m_sync, 0, EGL_FOREVER_KHR);
}

void GLFenceEGL::serverWait()
{
    if (!capabilities().eglServerWaitSupported) {
        clientWait();
        return;
    }

    auto& display = PlatformDisplay::sharedDisplay();
    if (display.eglCheckVersion(1, 5))
        eglWaitSync(display.eglDisplay(), m_sync, 0);
    else
        eglWaitSyncKHR(display.eglDisplay(), m_sync, 0);
}

#if OS(UNIX)
UnixFileDescriptor GLFenceEGL::exportFD()
{
    if (!m_isExportable)
        return { };

    return UnixFileDescriptor { eglDupNativeFenceFDANDROID(PlatformDisplay::sharedDisplay().eglDisplay(), m_sync), UnixFileDescriptor::Adopt };
}
#endif

} // namespace WebCore

#endif // HAVE(GL_FENCE)
