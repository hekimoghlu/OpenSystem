/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
#include "DragSource.h"

#if ENABLE(DRAG_SUPPORT) && USE(GTK4)

#include "WebKitWebViewBasePrivate.h"
#include <WebCore/GtkUtilities.h>
#include <WebCore/PasteboardCustomData.h>
#include <gtk/gtk.h>

namespace WebKit {
using namespace WebCore;

DragSource::DragSource(GtkWidget* webView)
    : m_webView(webView)
{
}

DragSource::~DragSource()
{
}

void DragSource::begin(SelectionData&& selectionData, OptionSet<DragOperation> operationMask, RefPtr<ShareableBitmap>&& image, IntPoint&& imageHotspot)
{
    if (m_drag) {
        gdk_drag_drop_done(m_drag.get(), FALSE);
        m_drag = nullptr;
    }

    m_selectionData = WTFMove(selectionData);

    Vector<GdkContentProvider*> providers;
    if (m_selectionData->hasMarkup()) {
        CString markup = m_selectionData->markup().utf8();
        GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new(markup.data(), markup.length()));
        providers.append(gdk_content_provider_new_for_bytes("text/html", bytes.get()));
    }

    if (m_selectionData->hasURIList()) {
        CString uriList = m_selectionData->uriList().utf8();
        GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new(uriList.data(), uriList.length()));
        providers.append(gdk_content_provider_new_for_bytes("text/uri-list", bytes.get()));
    }

    if (m_selectionData->hasURL()) {
        CString urlString = m_selectionData->url().string().utf8();
        gchar* url = g_strdup_printf("%s\n%s", urlString.data(), m_selectionData->hasText() ? m_selectionData->text().utf8().data() : urlString.data());
        GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new_take(url, strlen(url)));
        providers.append(gdk_content_provider_new_for_bytes("_NETSCAPE_URL", bytes.get()));
    }

    if (m_selectionData->hasImage()) {
        auto pixbuf = m_selectionData->image()->adapter().gdkPixbuf();
        providers.append(gdk_content_provider_new_typed(GDK_TYPE_PIXBUF, pixbuf.get()));
    }

    if (m_selectionData->hasText())
        providers.append(gdk_content_provider_new_typed(G_TYPE_STRING, m_selectionData->text().utf8().data()));

    if (m_selectionData->canSmartReplace()) {
        GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new(nullptr, 0));
        providers.append(gdk_content_provider_new_for_bytes("application/vnd.webkitgtk.smartpaste", bytes.get()));
    }

    if (m_selectionData->hasCustomData()) {
        GRefPtr<GBytes> bytes = m_selectionData->customData()->createGBytes();
        providers.append(gdk_content_provider_new_for_bytes(PasteboardCustomData::gtkType().characters(), bytes.get()));
    }

    auto* surface = gtk_native_get_surface(gtk_widget_get_native(m_webView));
    auto* device = gdk_seat_get_pointer(gdk_display_get_default_seat(gtk_widget_get_display(m_webView)));
    GRefPtr<GdkContentProvider> provider = adoptGRef(gdk_content_provider_new_union(providers.data(), providers.size()));
    m_drag = adoptGRef(gdk_drag_begin(surface, device, provider.get(), dragOperationToGdkDragActions(operationMask), 0, 0));
    g_signal_connect(m_drag.get(), "dnd-finished", G_CALLBACK(+[](GdkDrag* gtkDrag, gpointer userData) {
        auto& drag = *static_cast<DragSource*>(userData);
        if (drag.m_drag.get() != gtkDrag)
            return;

        drag.m_selectionData = std::nullopt;
        drag.m_drag = nullptr;

        GdkDevice* device = gdk_drag_get_device(gtkDrag);
        double x = 0;
        double y = 0;
        gdk_device_get_surface_at_position(device, &x, &y);

        auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(drag.m_webView));
        ASSERT(page);

        IntPoint point(x, y);
        page->dragEnded(point, point, gdkDragActionToDragOperation(gdk_drag_get_selected_action(gtkDrag)));
    }), this);

    g_signal_connect(m_drag.get(), "cancel", G_CALLBACK(+[](GdkDrag* gtkDrag, GdkDragCancelReason, gpointer userData) {
        auto& drag = *static_cast<DragSource*>(userData);
        if (drag.m_drag.get() != gtkDrag)
            return;

        drag.m_selectionData = std::nullopt;
        drag.m_drag = nullptr;
    }), this);

    auto* dragIcon = gtk_drag_icon_get_for_drag(m_drag.get());
    RefPtr<Image> iconImage = image ? image->createImage() : nullptr;
    if (iconImage) {
        if (auto texture = iconImage->adapter().gdkTexture()) {
            gdk_drag_set_hotspot(m_drag.get(), -imageHotspot.x(), -imageHotspot.y());
            auto* picture = gtk_picture_new_for_paintable(GDK_PAINTABLE(texture.get()));
            gtk_drag_icon_set_child(GTK_DRAG_ICON(dragIcon), picture);
            return;
        }
    }

    gdk_drag_set_hotspot(m_drag.get(), -2, -2);
    auto* child = gtk_image_new_from_icon_name("text-x-generic");
    gtk_image_set_icon_size(GTK_IMAGE(child), GTK_ICON_SIZE_LARGE);
    gtk_drag_icon_set_child(GTK_DRAG_ICON(dragIcon), child);
}

} // namespace WebKit

#endif // ENABLE(DRAG_SUPPORT) && USE(GTK4)
