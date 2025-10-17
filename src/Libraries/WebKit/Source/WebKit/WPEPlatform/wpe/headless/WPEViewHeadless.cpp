/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "WPEViewHeadless.h"

#include "WPEToplevelHeadless.h"
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/RunLoopSourcePriority.h>
#include <wtf/glib/WTFGType.h>

/**
 * WPEViewHeadless:
 *
 */
struct _WPEViewHeadlessPrivate {
    GRefPtr<WPEBuffer> pendingBuffer;
    GRefPtr<WPEBuffer> committedBuffer;
    GRefPtr<GSource> frameSource;
    gint64 lastFrameTime;
};
WEBKIT_DEFINE_FINAL_TYPE(WPEViewHeadless, wpe_view_headless, WPE_TYPE_VIEW, WPEView)

static GSourceFuncs frameSourceFuncs = {
    nullptr, // prepare
    nullptr, // check
    // dispatch
    [](GSource* source, GSourceFunc callback, gpointer userData) -> gboolean
    {
        if (g_source_get_ready_time(source) == -1)
            return G_SOURCE_CONTINUE;
        g_source_set_ready_time(source, -1);
        return callback(userData);
    },
    nullptr, // finalize
    nullptr, // closure_callback
    nullptr, // closure_marshall
};

static void wpeViewHeadlessConstructed(GObject* object)
{
    G_OBJECT_CLASS(wpe_view_headless_parent_class)->constructed(object);

    auto* view = WPE_VIEW(object);
    g_signal_connect(view, "notify::toplevel", G_CALLBACK(+[](WPEView* view, GParamSpec*, gpointer) {
        auto* toplevel = wpe_view_get_toplevel(view);
        if (!toplevel) {
            wpe_view_unmap(view);
            return;
        }

        int width;
        int height;
        wpe_toplevel_get_size(toplevel, &width, &height);
        if (width && height)
            wpe_view_resized(view, width, height);

        wpe_view_map(view);
    }), nullptr);

    auto* priv = WPE_VIEW_HEADLESS(view)->priv;
    priv->frameSource = adoptGRef(g_source_new(&frameSourceFuncs, sizeof(GSource)));
    g_source_set_priority(priv->frameSource.get(), RunLoopSourcePriority::RunLoopTimer);
    g_source_set_name(priv->frameSource.get(), "WPE headless frame timer");
    g_source_set_callback(priv->frameSource.get(), [](gpointer userData) -> gboolean {
        auto* view = WPE_VIEW(userData);
        auto* priv = WPE_VIEW_HEADLESS(view)->priv;
        if (priv->committedBuffer)
            wpe_view_buffer_released(view, priv->committedBuffer.get());
        priv->committedBuffer = WTFMove(priv->pendingBuffer);
        wpe_view_buffer_rendered(view, priv->committedBuffer.get());

        if (g_source_is_destroyed(priv->frameSource.get()))
            return G_SOURCE_REMOVE;
        return G_SOURCE_CONTINUE;
    }, object, nullptr);
    g_source_attach(priv->frameSource.get(), g_main_context_get_thread_default());
    g_source_set_ready_time(priv->frameSource.get(), -1);
}

static void wpeViewHeadlessDispose(GObject* object)
{
    auto* priv = WPE_VIEW_HEADLESS(object)->priv;

    if (priv->frameSource) {
        g_source_destroy(priv->frameSource.get());
        priv->frameSource = nullptr;
    }

    G_OBJECT_CLASS(wpe_view_headless_parent_class)->dispose(object);
}

static gboolean wpeViewHeadlessRenderBuffer(WPEView* view, WPEBuffer* buffer, const WPERectangle*, guint, GError**)
{
    auto* priv = WPE_VIEW_HEADLESS(view)->priv;
    priv->pendingBuffer = buffer;
    auto now = g_get_monotonic_time();
    if (!priv->lastFrameTime)
        priv->lastFrameTime = now;
    auto next = priv->lastFrameTime + (G_USEC_PER_SEC / 60);
    priv->lastFrameTime = now;
    if (next <= now)
        g_source_set_ready_time(priv->frameSource.get(), 0);
    else
        g_source_set_ready_time(priv->frameSource.get(), next);

    return TRUE;
}

static void wpe_view_headless_class_init(WPEViewHeadlessClass* viewHeadlessClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(viewHeadlessClass);
    objectClass->constructed = wpeViewHeadlessConstructed;
    objectClass->dispose = wpeViewHeadlessDispose;

    WPEViewClass* viewClass = WPE_VIEW_CLASS(viewHeadlessClass);
    viewClass->render_buffer = wpeViewHeadlessRenderBuffer;
}

/**
 * wpe_view_headless_new:
 * @display: a #WPEDisplayHeadless
 *
 * Create a new #WPEViewHeadless
 *
 * Returns: (transfer full): a #WPEView
 */
WPEView* wpe_view_headless_new(WPEDisplayHeadless* display)
{
    g_return_val_if_fail(WPE_IS_DISPLAY_HEADLESS(display), nullptr);

    return WPE_VIEW(g_object_new(WPE_TYPE_VIEW_HEADLESS, "display", display, nullptr));
}
