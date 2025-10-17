/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

#if USE(GSTREAMER)

#include "GLContext.h"
#include "GStreamerCommon.h"
#include <gst/gl/gl.h>
#if GST_GL_HAVE_PLATFORM_EGL
#include <gst/gl/egl/gstgldisplay_egl.h>
#endif
#include <wtf/glib/GUniquePtr.h>

GST_DEBUG_CATEGORY(webkit_display_debug);
#define GST_CAT_DEFAULT webkit_display_debug

namespace WebCore {

static void ensureDebugCategoryInitialized()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_display_debug, "webkitdisplay", 0, "WebKit Display");
    });
}

GstGLDisplay* PlatformDisplay::gstGLDisplay() const
{
    ensureDebugCategoryInitialized();
    if (!m_gstGLDisplay)
        m_gstGLDisplay = adoptGRef(GST_GL_DISPLAY(gst_gl_display_egl_new_with_egl_display(eglDisplay())));
    GST_TRACE("Using GL display %" GST_PTR_FORMAT, m_gstGLDisplay.get());
    return m_gstGLDisplay.get();
}

GstGLContext* PlatformDisplay::gstGLContext() const
{
    ensureDebugCategoryInitialized();

    if (m_gstGLContext)
        return m_gstGLContext.get();

    auto* gstDisplay = gstGLDisplay();
    if (!gstDisplay) {
        GST_ERROR("No GL display");
        return nullptr;
    }

    auto* context = const_cast<PlatformDisplay*>(this)->sharingGLContext();
    if (!context) {
        GST_ERROR("No sharing GL context");
        return nullptr;
    }

    m_gstGLContext = adoptGRef(gst_gl_context_new_wrapped(gstDisplay, reinterpret_cast<guintptr>(context->platformContext()), GST_GL_PLATFORM_EGL, GST_GL_API_GLES2));
    {
        GLContext::ScopedGLContextCurrent scopedCurrent(*context);
        if (gst_gl_context_activate(m_gstGLContext.get(), TRUE)) {
            GUniqueOutPtr<GError> error;
            if (!gst_gl_context_fill_info(m_gstGLContext.get(), &error.outPtr()))
                GST_ERROR("Failed to fill in GStreamer context: %s", error->message);
            gst_gl_context_activate(m_gstGLContext.get(), FALSE);
        }
    }
    GST_DEBUG("Created GL context %" GST_PTR_FORMAT, m_gstGLContext.get());
    return m_gstGLContext.get();
}

void PlatformDisplay::clearGStreamerGLState()
{
    m_gstGLDisplay = nullptr;
    m_gstGLContext = nullptr;
}

} // namespace WebCore

#undef GST_CAT_DEFAULT

#endif // USE(GSTREAMER)
