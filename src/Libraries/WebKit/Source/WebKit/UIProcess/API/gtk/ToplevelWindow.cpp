/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "ToplevelWindow.h"

#include "WebKitWebViewBasePrivate.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

static HashMap<GtkWindow*, std::unique_ptr<ToplevelWindow>>& toplevelWindows()
{
    static NeverDestroyed<HashMap<GtkWindow*, std::unique_ptr<ToplevelWindow>>> windows;
    return windows.get();
}

static void toplevelWindowDestroyedCallback(gpointer, GObject* window)
{
    toplevelWindows().remove(reinterpret_cast<GtkWindow*>(window));
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ToplevelWindow);

ToplevelWindow* ToplevelWindow::forGtkWindow(GtkWindow* window)
{
    auto addResult = toplevelWindows().ensure(window, [window] {
        return makeUnique<ToplevelWindow>(window);
    });
    if (addResult.isNewEntry)
        g_object_weak_ref(G_OBJECT(window), toplevelWindowDestroyedCallback, nullptr);

    return addResult.iterator->value.get();
}

ToplevelWindow::ToplevelWindow(GtkWindow* window)
    : m_window(window)
{
}

ToplevelWindow::~ToplevelWindow()
{
    ASSERT(m_webViews.isEmpty());
}

void ToplevelWindow::addWebView(WebKitWebViewBase* webView)
{
    auto addResult = m_webViews.add(webView);
    if (addResult.isNewEntry && m_webViews.size() == 1)
        connectSignals();
#if !USE(GTK4)
    if (gtk_widget_get_realized(GTK_WIDGET(m_window)))
        gtk_widget_realize(GTK_WIDGET(webView));
#endif
}

void ToplevelWindow::removeWebView(WebKitWebViewBase* webView)
{
    m_webViews.remove(webView);
    if (m_webViews.isEmpty()) {
        disconnectSignals();
        g_object_weak_unref(G_OBJECT(m_window), toplevelWindowDestroyedCallback, nullptr);
        toplevelWindows().remove(reinterpret_cast<GtkWindow*>(m_window));
    }
}

bool ToplevelWindow::isActive() const
{
    return gtk_widget_get_visible(GTK_WIDGET(m_window)) && gtk_window_is_active(m_window);
}

bool ToplevelWindow::isFullscreen() const
{
#if USE(GTK4)
    return m_state & GDK_TOPLEVEL_STATE_FULLSCREEN;
#else
    if (auto* window = gtk_widget_get_window(GTK_WIDGET(m_window)))
        return gdk_window_get_state(window) & GDK_WINDOW_STATE_FULLSCREEN;
    return false;
#endif
}

bool ToplevelWindow::isMinimized() const
{
#if USE(GTK4)
    return m_state & GDK_TOPLEVEL_STATE_MINIMIZED;
#else
    if (auto* window = gtk_widget_get_window(GTK_WIDGET(m_window)))
        return gdk_window_get_state(window) & GDK_WINDOW_STATE_ICONIFIED;
    return false;
#endif
}

bool ToplevelWindow::isSuspended() const
{
#if GTK_CHECK_VERSION(4, 12, 0)
    return m_state & GDK_TOPLEVEL_STATE_SUSPENDED;
#else
    return false;
#endif
}

GdkMonitor* ToplevelWindow::monitor() const
{
    auto* display = gtk_widget_get_display(GTK_WIDGET(m_window));
#if USE(GTK4)
    if (auto* surface = gtk_native_get_surface(GTK_NATIVE(m_window)))
        return gdk_display_get_monitor_at_surface(display, surface);
#else
    if (auto* window = gtk_widget_get_window(GTK_WIDGET(m_window)))
        return gdk_display_get_monitor_at_window(display, window);
#endif
    return nullptr;
}

bool ToplevelWindow::isInMonitor() const
{
#if USE(GTK4)
    // GTK4 always returns a valid monitor from gdk_display_get_monitor_at_surface() even after monitor-leave signal is emitted,
    // so we keep track of the monitors.
    return !m_monitors.isEmpty();
#else
    auto* widget = GTK_WIDGET(m_window);
    return gtk_widget_get_realized(widget) && gdk_display_get_monitor_at_window(gtk_widget_get_display(widget), gtk_widget_get_window(widget));
#endif
}

void ToplevelWindow::connectSignals()
{
#if USE(GTK4)
    g_signal_connect(m_window, "notify::is-active", G_CALLBACK(+[](GtkWindow* window, GParamSpec*, ToplevelWindow* toplevelWindow) {
        toplevelWindow->notifyIsActive(gtk_window_is_active(window));
    }), this);
    if (gtk_widget_get_realized(GTK_WIDGET(m_window)))
        connectSurfaceSignals();
    else {
        g_signal_connect_swapped(m_window, "realize", G_CALLBACK(+[](ToplevelWindow* toplevelWindow) {
            toplevelWindow->connectSurfaceSignals();
        }), this);
    }
    g_signal_connect_swapped(m_window, "unrealize", G_CALLBACK(+[](ToplevelWindow* toplevelWindow) {
        toplevelWindow->disconnectSurfaceSignals();
    }), this);
#else
    g_signal_connect(m_window, "focus-in-event", G_CALLBACK(+[](GtkWidget* widget, GdkEventFocus*, ToplevelWindow* toplevelWindow) -> gboolean {
        // Spurious focus in events can occur when the window is hidden.
        if (!gtk_widget_get_visible(widget))
            return FALSE;
        toplevelWindow->notifyIsActive(true);
        return FALSE;
    }), this);
    g_signal_connect(m_window, "focus-out-event", G_CALLBACK(+[](GtkWidget* widget, GdkEventFocus*, ToplevelWindow* toplevelWindow) -> gboolean {
        toplevelWindow->notifyIsActive(false);
        return FALSE;
    }), this);
    g_signal_connect(m_window, "window-state-event", G_CALLBACK(+[](GtkWidget*, GdkEventWindowState* event, ToplevelWindow* toplevelWindow) -> gboolean {
        toplevelWindow->notifyState(event->changed_mask, event->new_window_state);
        return FALSE;
    }), this);
    g_signal_connect(m_window, "configure-event", G_CALLBACK(+[](GtkWidget* widget, GdkEventConfigure*, ToplevelWindow* toplevelWindow) -> gboolean {
        if (!gtk_widget_get_realized(widget))
            return FALSE;
        toplevelWindow->notifyMonitorChanged(toplevelWindow->monitor());
        return FALSE;
    }), this);
    if (!gtk_widget_get_realized(GTK_WIDGET(m_window))) {
        g_signal_connect_swapped(m_window, "realize", G_CALLBACK(+[](ToplevelWindow* toplevelWindow) {
            for (auto* webView : toplevelWindow->m_webViews)
                gtk_widget_realize(GTK_WIDGET(webView));
        }), this);
    }
#endif
}

void ToplevelWindow::disconnectSignals()
{
    g_signal_handlers_disconnect_by_data(m_window, this);
#if USE(GTK4)
    if (gtk_widget_get_realized(GTK_WIDGET(m_window)))
        disconnectSurfaceSignals();
#endif
}

#if USE(GTK4)
void ToplevelWindow::connectSurfaceSignals()
{
    auto* surface = gtk_native_get_surface(GTK_NATIVE(m_window));
    m_state = gdk_toplevel_get_state(GDK_TOPLEVEL(surface));
    g_signal_connect(surface, "notify::state", G_CALLBACK(+[](GdkSurface* surface, GParamSpec*, ToplevelWindow* toplevelWindow) {
        auto state = gdk_toplevel_get_state(GDK_TOPLEVEL(surface));
        auto changedMask = toplevelWindow->m_state ^ state;
        toplevelWindow->m_state = state;
        toplevelWindow->notifyState(changedMask, state);
    }), this);
    if (auto* monitor = gdk_display_get_monitor_at_surface(gtk_widget_get_display(GTK_WIDGET(m_window)), surface)) {
        m_monitors.add(monitor);
        notifyMonitorChanged(monitor);
    }
    g_signal_connect(surface, "enter-monitor", G_CALLBACK(+[](GdkSurface* surface, GdkMonitor* monitor, ToplevelWindow* toplevelWindow) {
        toplevelWindow->m_monitors.add(monitor);
        toplevelWindow->notifyMonitorChanged(monitor);
    }), this);
    g_signal_connect(surface, "leave-monitor", G_CALLBACK(+[](GdkSurface* surface, GdkMonitor* monitor, ToplevelWindow* toplevelWindow) {
        toplevelWindow->m_monitors.remove(monitor);
        toplevelWindow->notifyMonitorChanged(toplevelWindow->monitor());
    }), this);
}

void ToplevelWindow::disconnectSurfaceSignals()
{
    auto* surface = gtk_native_get_surface(GTK_NATIVE(m_window));
    g_signal_handlers_disconnect_by_data(surface, this);
    m_state = static_cast<GdkToplevelState>(0);
}
#endif

void ToplevelWindow::notifyIsActive(bool isActive)
{
    for (auto* webView : m_webViews)
        webkitWebViewBaseToplevelWindowIsActiveChanged(webView, isActive);
}

void ToplevelWindow::notifyState(uint32_t mask, uint32_t state)
{
    for (auto* webView : m_webViews)
        webkitWebViewBaseToplevelWindowStateChanged(webView, mask, state);
}

void ToplevelWindow::notifyMonitorChanged(GdkMonitor* monitor)
{
    for (auto* webView : m_webViews) {
#if !USE(GTK4)
        if (!gtk_widget_get_realized(GTK_WIDGET(webView)))
            continue;
#endif
        webkitWebViewBaseToplevelWindowMonitorChanged(webView, monitor);
    }
}

} // namespace WebKit
