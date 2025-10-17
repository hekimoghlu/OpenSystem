/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
#include "WPEDisplayQtQuick.h"

#include "WPEToplevelQtQuick.h"
#include "WPEViewQtQuick.h"

#include <epoxy/egl.h>

#include <QGuiApplication>
#include <qpa/qplatformnativeinterface.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/WTFGType.h>
#include <wtf/text/CString.h>

#ifndef EGL_DRM_RENDER_NODE_FILE_EXT
#define EGL_DRM_RENDER_NODE_FILE_EXT 0x3377
#endif

/**
 * WPEDisplayQtQuick:
 *
 */
struct _WPEDisplayQtQuickPrivate {
    EGLDisplay eglDisplay;
    CString drmDevice;
    CString drmRenderNode;
};
WEBKIT_DEFINE_FINAL_TYPE(WPEDisplayQtQuick, wpe_display_qtquick, WPE_TYPE_DISPLAY, WPEDisplay)

static gboolean wpeDisplayQtQuickConnect(WPEDisplay* display, GError** error)
{
    auto* priv = WPE_DISPLAY_QTQUICK(display)->priv;

    auto eglDisplay = static_cast<EGLDisplay>(QGuiApplication::platformNativeInterface()->nativeResourceForIntegration("eglDisplay"));
    if (!eglDisplay) {
        g_set_error_literal(error, WPE_VIEW_ERROR, WPE_VIEW_ERROR_RENDER_FAILED, "Failed to initialize rendering: Cannot access EGL display via Qt");
        return FALSE;
    }

    priv->eglDisplay = eglDisplay;

    if (!epoxy_has_egl_extension(eglDisplay, "EGL_EXT_device_query")) {
        g_set_error_literal(error, WPE_VIEW_ERROR, WPE_VIEW_ERROR_RENDER_FAILED, "Failed to initialize rendering: 'EGL_EXT_device_query' not available");
        return FALSE;
    }

    EGLDeviceEXT eglDevice;
    if (!eglQueryDisplayAttribEXT(eglDisplay, EGL_DEVICE_EXT, reinterpret_cast<EGLAttrib*>(&eglDevice))) {
        g_set_error_literal(error, WPE_VIEW_ERROR, WPE_VIEW_ERROR_RENDER_FAILED, "Failed to initialize rendering: 'EGLDeviceEXT' not available");
        return FALSE;
    }

    const char* extensions = eglQueryDeviceStringEXT(eglDevice, EGL_EXTENSIONS);
    if (epoxy_extension_in_string(extensions, "EGL_EXT_device_drm"))
        priv->drmDevice = eglQueryDeviceStringEXT(eglDevice, EGL_DRM_DEVICE_FILE_EXT);
    else {
        g_set_error_literal(error, WPE_VIEW_ERROR, WPE_VIEW_ERROR_RENDER_FAILED, "Failed to initialize rendering: 'EGL_EXT_device_drm' not available");
        return FALSE;
    }

    if (epoxy_extension_in_string(extensions, "EGL_EXT_device_drm_render_node"))
        priv->drmRenderNode = eglQueryDeviceStringEXT(eglDevice, EGL_DRM_RENDER_NODE_FILE_EXT);
    else {
        g_set_error_literal(error, WPE_VIEW_ERROR, WPE_VIEW_ERROR_RENDER_FAILED, "Failed to initialize rendering: 'EGL_EXT_device_drm_render_node' not available");
        return FALSE;
    }

    return TRUE;
}

static WPEView* wpeDisplayQtQuickCreateView(WPEDisplay* display)
{
    auto* displayQt = WPE_DISPLAY_QTQUICK(display);
    auto* view = wpe_view_qtquick_new(displayQt);

    GRefPtr<WPEToplevel> toplevel = adoptGRef(wpe_toplevel_qtquick_new(displayQt));
    wpe_view_set_toplevel(view, toplevel.get());

    return view;
}

static gpointer wpeDisplayQtQuickGetEGLDisplay(WPEDisplay* display, GError**)
{
    return WPE_DISPLAY_QTQUICK(display)->priv->eglDisplay;
}

static const char* wpeDisplayQtQuickGetDRMDevice(WPEDisplay* display)
{
    return WPE_DISPLAY_QTQUICK(display)->priv->drmDevice.data();
}

static const char* wpeDisplayQtQuickGetDRMRenderNode(WPEDisplay* display)
{
    auto* priv = WPE_DISPLAY_QTQUICK(display)->priv;
    if (!priv->drmRenderNode.isNull())
        return priv->drmRenderNode.data();
    return priv->drmDevice.data();
}

static void wpe_display_qtquick_class_init(WPEDisplayQtQuickClass* displayQtQuickClass)
{
    WPEDisplayClass* displayClass = WPE_DISPLAY_CLASS(displayQtQuickClass);
    displayClass->connect = wpeDisplayQtQuickConnect;
    displayClass->create_view = wpeDisplayQtQuickCreateView;
    displayClass->get_egl_display = wpeDisplayQtQuickGetEGLDisplay;
    displayClass->get_drm_device = wpeDisplayQtQuickGetDRMDevice;
    displayClass->get_drm_render_node = wpeDisplayQtQuickGetDRMRenderNode;
}

WPEDisplay* wpe_display_qtquick_new(void)
{
    return WPE_DISPLAY(g_object_new(WPE_TYPE_DISPLAY_QTQUICK, nullptr));
}
