/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#include "DRMDevice.h"

#include <WebCore/GLContext.h>
#include <WebCore/PlatformDisplay.h>
#include <epoxy/egl.h>
#include <mutex>
#include <wtf/Function.h>
#include <wtf/NeverDestroyed.h>

#if PLATFORM(GTK)
#include "Display.h"
#endif

#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
#include <wpe/wpe-platform.h>
#endif

#if USE(LIBDRM)
#include <xf86drm.h>
#endif

#ifndef EGL_DRM_RENDER_NODE_FILE_EXT
#define EGL_DRM_RENDER_NODE_FILE_EXT 0x3377
#endif

namespace WebKit {

static EGLDeviceEXT eglDisplayDevice(EGLDisplay eglDisplay)
{
    if (!WebCore::GLContext::isExtensionSupported(eglQueryString(nullptr, EGL_EXTENSIONS), "EGL_EXT_device_query"))
        return nullptr;

    EGLDeviceEXT eglDevice;
    if (eglQueryDisplayAttribEXT(eglDisplay, EGL_DEVICE_EXT, reinterpret_cast<EGLAttrib*>(&eglDevice)))
        return eglDevice;

    return nullptr;
}

#if USE(LIBDRM)
static void drmForeachDevice(Function<bool(drmDevice*)>&& functor)
{
    std::array<drmDevicePtr, 64> devices = { };

    int numDevices = drmGetDevices2(0, devices.data(), devices.size());
    if (numDevices <= 0)
        return;

    for (int i = 0; i < numDevices; ++i) {
        if (!functor(devices[i]))
            break;
    }
    drmFreeDevices(devices.data(), numDevices);
}

static String drmFirstRenderNode()
{
    String renderNodeDeviceFile;
    drmForeachDevice([&](drmDevice* device) {
        if (!(device->available_nodes & (1 << DRM_NODE_RENDER)))
            return true;

        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE Port
        renderNodeDeviceFile = String::fromUTF8(device->nodes[DRM_NODE_RENDER]);
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        return false;
    });
    return renderNodeDeviceFile;
}

static String drmRenderNodeFromPrimaryDeviceFile(const String& primaryDeviceFile)
{
    if (primaryDeviceFile.isEmpty())
        return drmFirstRenderNode();

    String renderNodeDeviceFile;
    drmForeachDevice([&](drmDevice* device) {
        if (!(device->available_nodes & (1 << DRM_NODE_PRIMARY | 1 << DRM_NODE_RENDER)))
            return true;

        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE Port
        if (String::fromUTF8(device->nodes[DRM_NODE_PRIMARY]) == primaryDeviceFile) {
            renderNodeDeviceFile = String::fromUTF8(device->nodes[DRM_NODE_RENDER]);
            return false;
        }
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

        return true;
    });
    // If we fail to find a render node for the device file, just use the device file as render node.
    return !renderNodeDeviceFile.isEmpty() ? renderNodeDeviceFile : primaryDeviceFile;
}
#endif

static String drmRenderNodeForEGLDisplay(EGLDisplay eglDisplay)
{
    if (EGLDeviceEXT device = eglDisplayDevice(eglDisplay)) {
        if (WebCore::GLContext::isExtensionSupported(eglQueryDeviceStringEXT(device, EGL_EXTENSIONS), "EGL_EXT_device_drm_render_node"))
            return String::fromUTF8(eglQueryDeviceStringEXT(device, EGL_DRM_RENDER_NODE_FILE_EXT));

#if USE(LIBDRM)
        // If EGL_EXT_device_drm_render_node is not present, try to get the render node using DRM API.
        return drmRenderNodeFromPrimaryDeviceFile(drmPrimaryDevice());
#endif
    }

#if USE(LIBDRM)
    // If EGLDevice is not available, just get the first render node returned by DRM.
    return drmFirstRenderNode();
#else
    return { };
#endif
}

static String drmPrimaryDeviceForEGLDisplay(EGLDisplay eglDisplay)
{
    EGLDeviceEXT device = eglDisplayDevice(eglDisplay);
    if (!device)
        return { };

    if (!WebCore::GLContext::isExtensionSupported(eglQueryDeviceStringEXT(device, EGL_EXTENSIONS), "EGL_EXT_device_drm"))
        return { };

    return String::fromUTF8(eglQueryDeviceStringEXT(device, EGL_DRM_DEVICE_FILE_EXT));
}

static EGLDisplay currentEGLDisplay()
{
#if PLATFORM(GTK)
    if (auto* glDisplay = Display::singleton().glDisplay())
        return glDisplay->eglDisplay();
#endif

    auto eglDisplay = eglGetCurrentDisplay();
    if (eglDisplay != EGL_NO_DISPLAY)
        return eglDisplay;

    return eglGetDisplay(EGL_DEFAULT_DISPLAY);
}

const String& drmPrimaryDevice()
{
    static LazyNeverDestroyed<String> primaryDevice;
    static std::once_flag once;
    std::call_once(once, [] {
#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
        if (g_type_class_peek(WPE_TYPE_DISPLAY)) {
            primaryDevice.construct(String::fromUTF8(wpe_display_get_drm_device(wpe_display_get_primary())));
            return;
        }
#endif

        auto eglDisplay = currentEGLDisplay();
        if (eglDisplay != EGL_NO_DISPLAY) {
            primaryDevice.construct(drmPrimaryDeviceForEGLDisplay(eglDisplay));
            return;
        }

        primaryDevice.construct();
    });
    return primaryDevice.get();
}

const String& drmRenderNodeDevice()
{
    static LazyNeverDestroyed<String> renderNodeDevice;
    static std::once_flag once;
    std::call_once(once, [] {
#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
        if (g_type_class_peek(WPE_TYPE_DISPLAY)) {
            renderNodeDevice.construct(String::fromUTF8(wpe_display_get_drm_render_node(wpe_display_get_primary())));
            return;
        }
#endif

        const char* envDeviceFile = getenv("WEBKIT_WEB_RENDER_DEVICE_FILE");
        if (envDeviceFile && *envDeviceFile) {
            renderNodeDevice.construct(String::fromUTF8(envDeviceFile));
            return;
        }

        auto eglDisplay = currentEGLDisplay();
        if (eglDisplay != EGL_NO_DISPLAY) {
            renderNodeDevice.construct(drmRenderNodeForEGLDisplay(eglDisplay));
            return;
        }

        renderNodeDevice.construct();
    });
    return renderNodeDevice.get();
}

} // namespace WebKit
