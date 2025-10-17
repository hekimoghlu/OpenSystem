/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#if ENABLE(DRAG_SUPPORT) && !USE(GTK4)

#include "WebKitWebViewBasePrivate.h"
#include <WebCore/GRefPtrGtk.h>
#include <WebCore/GdkSkiaUtilities.h>
#include <WebCore/GtkUtilities.h>
#include <WebCore/PasteboardCustomData.h>
#include <gtk/gtk.h>

namespace WebKit {
using namespace WebCore;

enum DragTargetType { Markup, Text, Image, URIList, NetscapeURL, SmartPaste, Custom };

DragSource::DragSource(GtkWidget* webView)
    : m_webView(webView)
{
    g_signal_connect(m_webView, "drag-data-get", G_CALLBACK(+[](GtkWidget*, GdkDragContext* context, GtkSelectionData* data, guint info, guint, gpointer userData) {
        auto& drag = *static_cast<DragSource*>(userData);
        if (drag.m_drag.get() != context)
            return;

        switch (info) {
        case DragTargetType::Text:
            gtk_selection_data_set_text(data, drag.m_selectionData->text().utf8().data(), -1);
            break;
        case DragTargetType::Markup: {
            CString markup = drag.m_selectionData->markup().utf8();
            gtk_selection_data_set(data, gdk_atom_intern_static_string("text/html"), 8, reinterpret_cast<const guchar*>(markup.data()), markup.length());
            break;
        }
        case DragTargetType::URIList: {
            CString uriList = drag.m_selectionData->uriList().utf8();
            gtk_selection_data_set(data, gdk_atom_intern_static_string("text/uri-list"), 8, reinterpret_cast<const guchar*>(uriList.data()), uriList.length());
            break;
        }
        case DragTargetType::NetscapeURL: {
            CString urlString = drag.m_selectionData->url().string().utf8();
            GUniquePtr<gchar> url(g_strdup_printf("%s\n%s", urlString.data(), drag.m_selectionData->hasText() ? drag.m_selectionData->text().utf8().data() : urlString.data()));
            gtk_selection_data_set(data, gdk_atom_intern_static_string("_NETSCAPE_URL"), 8, reinterpret_cast<const guchar*>(url.get()), strlen(url.get()));
            break;
        }
        case DragTargetType::Image: {
            auto pixbuf = drag.m_selectionData->image()->adapter().gdkPixbuf();
            gtk_selection_data_set_pixbuf(data, pixbuf.get());
            break;
        }
        case DragTargetType::SmartPaste:
            gtk_selection_data_set_text(data, "", -1);
            break;
        case DragTargetType::Custom: {
            auto buffer = drag.m_selectionData->customData()->span();
            gtk_selection_data_set(data, gdk_atom_intern_static_string(PasteboardCustomData::gtkType()), 8, reinterpret_cast<const guchar*>(buffer.data()), buffer.size());
            break;
        }
        }
    }), this);

    g_signal_connect(m_webView, "drag-end", G_CALLBACK(+[](GtkWidget*, GdkDragContext* context, gpointer userData) {
        auto& drag = *static_cast<DragSource*>(userData);
        if (drag.m_drag.get() != context)
            return;
        if (!drag.m_selectionData)
            return;

        drag.m_selectionData = std::nullopt;
        drag.m_drag = nullptr;

        GdkDevice* device = gdk_drag_context_get_device(context);
        int x = 0;
        int y = 0;
        gdk_device_get_window_at_position(device, &x, &y);
        int xRoot = 0;
        int yRoot = 0;
        gdk_device_get_position(device, nullptr, &xRoot, &yRoot);

        auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(drag.m_webView));
        ASSERT(page);
        page->dragEnded({ x, y }, { xRoot, yRoot }, gdkDragActionToDragOperation(gdk_drag_context_get_selected_action(context)));
    }), this);
}

DragSource::~DragSource()
{
    g_signal_handlers_disconnect_by_data(m_webView, this);
}

void DragSource::begin(SelectionData&& selectionData, OptionSet<DragOperation> operationMask, RefPtr<ShareableBitmap>&& image, IntPoint&& imageHotspot)
{
    if (m_drag) {
        gtk_drag_cancel(m_drag.get());
        m_drag = nullptr;
    }

    m_selectionData = WTFMove(selectionData);

    GRefPtr<GtkTargetList> list = adoptGRef(gtk_target_list_new(nullptr, 0));
    if (m_selectionData->hasText())
        gtk_target_list_add_text_targets(list.get(), DragTargetType::Text);
    if (m_selectionData->hasMarkup())
        gtk_target_list_add(list.get(), gdk_atom_intern_static_string("text/html"), 0, DragTargetType::Markup);
    if (m_selectionData->hasURIList())
        gtk_target_list_add_uri_targets(list.get(), DragTargetType::URIList);
    if (m_selectionData->hasURL())
        gtk_target_list_add(list.get(), gdk_atom_intern_static_string("_NETSCAPE_URL"), 0, DragTargetType::NetscapeURL);
    if (m_selectionData->hasImage())
        gtk_target_list_add_image_targets(list.get(), DragTargetType::Image, TRUE);
    if (m_selectionData->canSmartReplace())
        gtk_target_list_add(list.get(), gdk_atom_intern_static_string("application/vnd.webkitgtk.smartpaste"), 0, DragTargetType::SmartPaste);
    if (m_selectionData->hasCustomData())
        gtk_target_list_add(list.get(), gdk_atom_intern_static_string(PasteboardCustomData::gtkType()), 0, DragTargetType::Custom);

    m_drag = gtk_drag_begin_with_coordinates(m_webView, list.get(), dragOperationToGdkDragActions(operationMask), GDK_BUTTON_PRIMARY, nullptr, -1, -1);
    if (image) {
#if USE(CAIRO)
        RefPtr<cairo_surface_t> imageSurface(image->createCairoSurface());
#else
        auto skiaImage = image->createPlatformImage();
        RefPtr<cairo_surface_t> imageSurface(skiaImageToCairoSurface(*skiaImage));
#endif
        cairo_surface_set_device_offset(imageSurface.get(), -imageHotspot.x(), -imageHotspot.y());
        gtk_drag_set_icon_surface(m_drag.get(), imageSurface.get());
    } else
        gtk_drag_set_icon_default(m_drag.get());
}

} // namespace WebKit

#endif // ENABLE(DRAG_SUPPORT) && !USE(GTK4)
