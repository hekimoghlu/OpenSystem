/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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
#include "DropTarget.h"

#if ENABLE(DRAG_SUPPORT) && !USE(GTK4)

#include "SandboxExtension.h"
#include "WebKitWebViewBasePrivate.h"
#include <WebCore/DragData.h>
#include <WebCore/GRefPtrGtk.h>
#include <WebCore/GtkUtilities.h>
#include <WebCore/PasteboardCustomData.h>
#include <gtk/gtk.h>
#include <wtf/glib/GUniquePtr.h>

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::DropTarget> : std::true_type { };
}

namespace WebKit {
using namespace WebCore;

enum DropTargetType { Markup, Text, URIList, NetscapeURL, SmartPaste, Custom };

DropTarget::DropTarget(GtkWidget* webView)
    : m_webView(webView)
    , m_leaveTimer(RunLoop::main(), this, &DropTarget::leaveTimerFired)
{
    GRefPtr<GtkTargetList> list = adoptGRef(gtk_target_list_new(nullptr, 0));
    gtk_target_list_add_text_targets(list.get(), DropTargetType::Text);
    gtk_target_list_add(list.get(), gdk_atom_intern_static_string("text/html"), 0, DropTargetType::Markup);
    gtk_target_list_add_uri_targets(list.get(), DropTargetType::URIList);
    gtk_target_list_add(list.get(), gdk_atom_intern_static_string("_NETSCAPE_URL"), 0, DropTargetType::NetscapeURL);
    gtk_target_list_add(list.get(), gdk_atom_intern_static_string("application/vnd.webkitgtk.smartpaste"), 0, DropTargetType::SmartPaste);
    gtk_target_list_add(list.get(), gdk_atom_intern_static_string(PasteboardCustomData::gtkType().characters()), 0, DropTargetType::Custom);
    gtk_drag_dest_set(m_webView, static_cast<GtkDestDefaults>(0), nullptr, 0,
        static_cast<GdkDragAction>(GDK_ACTION_COPY | GDK_ACTION_MOVE | GDK_ACTION_LINK));
    gtk_drag_dest_set_target_list(m_webView, list.get());

    g_signal_connect_after(m_webView, "drag-motion", G_CALLBACK(+[](GtkWidget*, GdkDragContext* context, gint x, gint y, guint time, gpointer userData) -> gboolean {
        auto& drop = *static_cast<DropTarget*>(userData);
        if (drop.m_drop != context) {
            drop.accept(context, IntPoint(x, y), time);
        } else if (drop.m_drop == context)
            drop.update({ x, y }, time);
        return TRUE;
    }), this);

    g_signal_connect_after(m_webView, "drag-leave", G_CALLBACK(+[](GtkWidget*, GdkDragContext* context, guint time, gpointer userData) {
        auto& drop = *static_cast<DropTarget*>(userData);
        if (drop.m_drop != context)
            return;
        drop.leave();
    }), this);

    g_signal_connect_after(m_webView, "drag-drop", G_CALLBACK(+[](GtkWidget*, GdkDragContext* context, gint x, gint y, guint time, gpointer userData) -> gboolean {
        auto& drop = *static_cast<DropTarget*>(userData);
        if (drop.m_drop != context) {
            gtk_drag_finish(context, FALSE, FALSE, time);
            return TRUE;
        }
        drop.drop({ x, y }, time);
        return TRUE;
    }), this);

    g_signal_connect_after(m_webView, "drag-data-received", G_CALLBACK(+[](GtkWidget*, GdkDragContext* context, gint x, gint y, GtkSelectionData* data, guint info, guint time, gpointer userData) {
        auto& drop = *static_cast<DropTarget*>(userData);
        if (drop.m_drop != context)
            return;
        drop.dataReceived({ x, y }, data, info, time);
    }), this);
}

DropTarget::~DropTarget()
{
    g_signal_handlers_disconnect_by_data(m_webView, this);
}

void DropTarget::accept(GdkDragContext* drop, std::optional<WebCore::IntPoint> position, unsigned time)
{
    if (m_leaveTimer.isActive()) {
        m_leaveTimer.stop();
        leaveTimerFired();
    }

    m_drop = drop;
    m_position = position;
    m_dataRequestCount = 0;
    m_selectionData = std::nullopt;

    // WebCore needs the selection data to decide, so we need to preload the
    // data of targets we support. Once all data requests are done we start
    // notifying the web process about the DND events.
    auto* list = gdk_drag_context_list_targets(m_drop.get());
    static const char* const supportedTargets[] = {
        "text/plain;charset=utf-8",
        "text/html",
        "_NETSCAPE_URL",
        "text/uri-list",
        "application/vnd.webkitgtk.smartpaste",
        "org.webkitgtk.WebKit.custom-pasteboard-data"
    };
    Vector<GdkAtom, 4> targets;
    for (unsigned i = 0; i < G_N_ELEMENTS(supportedTargets); ++i) {
        GdkAtom atom = gdk_atom_intern_static_string(supportedTargets[i]);
        if (g_list_find(list, atom))
            targets.append(atom);
        else if (!i) {
            atom = gdk_atom_intern_static_string("text/plain");
            if (g_list_find(list, atom))
                targets.append(atom);
        }
    }

    if (targets.isEmpty())
        return;

    m_dataRequestCount = targets.size();
    m_selectionData = SelectionData();
    for (auto* atom : targets)
        gtk_drag_get_data(m_webView, m_drop.get(), atom, time);
}

void DropTarget::enter(IntPoint&& position, unsigned time)
{
    m_position = WTFMove(position);

    auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView));
    ASSERT(page);
    page->resetCurrentDragInformation();

    DragData dragData(&m_selectionData.value(), *m_position, convertWidgetPointToScreenPoint(m_webView, *m_position), gdkDragActionToDragOperation(gdk_drag_context_get_actions(m_drop.get())));
    page->dragEntered(dragData);
}

void DropTarget::update(IntPoint&& position, unsigned time)
{
    if (m_dataRequestCount || !m_selectionData)
        return;

    m_position = WTFMove(position);

    auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView));
    ASSERT(page);

    DragData dragData(&m_selectionData.value(), *m_position, convertWidgetPointToScreenPoint(m_webView, *m_position), gdkDragActionToDragOperation(gdk_drag_context_get_actions(m_drop.get())));
    page->dragUpdated(dragData);
}

void DropTarget::dataReceived(IntPoint&& position, GtkSelectionData* data, unsigned info, unsigned time)
{
    switch (info) {
    case DropTargetType::Text: {
        GUniquePtr<char> text(reinterpret_cast<char*>(gtk_selection_data_get_text(data)));
        m_selectionData->setText(String::fromUTF8(text.get()));
        break;
    }
    case DropTargetType::Markup: {
        gint length;
        const auto* markupData = gtk_selection_data_get_data_with_length(data, &length);
        if (length > 0) {
            // If data starts with UTF-16 BOM assume it's UTF-16, otherwise assume UTF-8.
            if (length >= 2 && reinterpret_cast<const UChar*>(markupData)[0] == 0xFEFF)
                m_selectionData->setMarkup(String({ reinterpret_cast<const UChar*>(markupData) + 1, static_cast<size_t>((length / 2) - 1) }));
            else
                m_selectionData->setMarkup(String::fromUTF8(std::span(markupData, length)));
        }
        break;
    }
    case DropTargetType::URIList: {
        gint length;
        const auto* uriListData = gtk_selection_data_get_data_with_length(data, &length);
        if (length > 0)
            m_selectionData->setURIList(String::fromUTF8(std::span(uriListData, length)));
        break;
    }
    case DropTargetType::NetscapeURL: {
        gint length;
        const auto* urlData = gtk_selection_data_get_data_with_length(data, &length);
        if (length > 0) {
            Vector<String> tokens = String::fromUTF8(std::span(urlData, length)).split('\n');
            URL url({ }, tokens[0]);
            if (url.isValid())
                m_selectionData->setURL(url, tokens.size() > 1 ? tokens[1] : String());
        }
        break;
    }
    case DropTargetType::SmartPaste:
        m_selectionData->setCanSmartReplace(true);
        break;
    case DropTargetType::Custom: {
        int length;
        const auto* customData = gtk_selection_data_get_data_with_length(data, &length);
        if (length > 0)
            m_selectionData->setCustomData(SharedBuffer::create(std::span { customData, static_cast<size_t>(length) }));
        break;
    }
    }

    if (--m_dataRequestCount)
        return;

    enter(WTFMove(position), time);
}

void DropTarget::didPerformAction()
{
    if (!m_drop)
        return;

    auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView));
    ASSERT(page);

    m_operation = page->currentDragOperation();

    gdk_drag_status(m_drop.get(), dragOperationToSingleGdkDragAction(m_operation), GDK_CURRENT_TIME);
}

void DropTarget::leaveTimerFired()
{
    auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView));
    ASSERT(page);

    SelectionData emptyData;
    DragData dragData(m_selectionData ? &m_selectionData.value() : &emptyData, *m_position, convertWidgetPointToScreenPoint(m_webView, *m_position), { });
    page->dragExited(dragData);
    page->resetCurrentDragInformation();

    m_drop = nullptr;
    m_position = std::nullopt;
    m_selectionData = std::nullopt;
}

void DropTarget::leave()
{
    // GTK emits drag-leave before drag-drop, but we need to know whether a drop
    // happened to notify the web process about the drag exit.
    m_leaveTimer.startOneShot(0_s);
}

void DropTarget::drop(IntPoint&& position, unsigned time)
{
    // If we don't have data at this point, allow the leave timer to fire, ending the drop operation.
    if (!m_selectionData)
        return;

    if (m_leaveTimer.isActive())
        m_leaveTimer.stop();

    auto* page = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView));
    ASSERT(page);

    OptionSet<DragApplicationFlags> flags;
    if (gdk_drag_context_get_selected_action(m_drop.get()) == GDK_ACTION_COPY)
        flags.add(DragApplicationFlags::IsCopyKeyDown);
    DragData dragData(&m_selectionData.value(), position, convertWidgetPointToScreenPoint(m_webView, position), gdkDragActionToDragOperation(gdk_drag_context_get_actions(m_drop.get())), flags);
    page->performDragOperation(dragData, { }, { }, { });
    gtk_drag_finish(m_drop.get(), TRUE, FALSE, time);

    m_drop = nullptr;
    m_position = std::nullopt;
    m_selectionData = std::nullopt;
}

} // namespace WebKit

#endif // ENABLE(DRAG_SUPPORT) && !USE(GTK4)
