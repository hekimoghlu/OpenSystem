/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
#include "SleepDisablerGLib.h"

#include <gio/gio.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/Sandbox.h>

namespace PAL {

std::unique_ptr<SleepDisabler> SleepDisabler::create(const String& reason, Type type)
{
    return std::unique_ptr<SleepDisabler>(new SleepDisablerGLib(reason, type));
}

SleepDisablerGLib::SleepDisablerGLib(const String& reason, Type type)
    : SleepDisabler(reason, type)
    , m_cancellable(adoptGRef(g_cancellable_new()))
    , m_reason(reason)
{
    // We ignore Type because we always want to inhibit both screen lock and
    // suspend, but only when idle. There is no reason for WebKit to ever block
    // a user from manually suspending the computer, so inhibiting idle
    // suffices. There's also probably no good reason for code taking a sleep
    // disabler to differentiate between lock and suspend on our platform. If we
    // ever need this distinction, which seems unlikely, then we'll need to
    // audit all use of SleepDisabler.

    const char* busName = shouldUsePortal() ? "org.freedesktop.portal.Desktop" : "org.freedesktop.ScreenSaver";
    const char* objectPath = shouldUsePortal() ? "/org/freedesktop/portal/desktop" : "/org/freedesktop/ScreenSaver";
    const char* interfaceName = shouldUsePortal() ? "org.freedesktop.portal.Inhibit" : "org.freedesktop.ScreenSaver";
    g_dbus_proxy_new_for_bus(G_BUS_TYPE_SESSION, static_cast<GDBusProxyFlags>(G_DBUS_PROXY_FLAGS_DO_NOT_LOAD_PROPERTIES | G_DBUS_PROXY_FLAGS_DO_NOT_CONNECT_SIGNALS),
        nullptr, busName, objectPath, interfaceName, m_cancellable.get(), [](GObject*, GAsyncResult* result, gpointer userData) {
        GUniqueOutPtr<GError> error;
        GRefPtr<GDBusProxy> proxy = adoptGRef(g_dbus_proxy_new_for_bus_finish(result, &error.outPtr()));
        if (g_error_matches(error.get(), G_IO_ERROR, G_IO_ERROR_CANCELLED))
            return;

        auto* self = static_cast<SleepDisablerGLib*>(userData);
        if (proxy) {
            GUniquePtr<char> nameOwner(g_dbus_proxy_get_name_owner(proxy.get()));
            if (nameOwner) {
                self->m_screenSaverProxy = WTFMove(proxy);
                self->acquireInhibitor();
                return;
            }
        }

        // Give up. Don't warn the user: this is expected.
        self->m_cancellable = nullptr;
    }, this);
}

SleepDisablerGLib::~SleepDisablerGLib()
{
    if (m_cancellable)
        g_cancellable_cancel(m_cancellable.get());
    else if (m_screenSaverCookie || m_inhibitPortalRequestObjectPath)
        releaseInhibitor();
}

void SleepDisablerGLib::acquireInhibitor()
{
    GVariant* parameters;
    if (shouldUsePortal()) {
        GVariantBuilder builder;
        g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);
        g_variant_builder_add(&builder, "{sv}", "reason", g_variant_new_string(m_reason.utf8().data()));
        parameters = g_variant_new("(su@a{sv})", "" /* no window */, 8 /* idle */, g_variant_builder_end(&builder));
    } else
        parameters = g_variant_new("(ss)", g_get_prgname(), m_reason.utf8().data());

    g_dbus_proxy_call(m_screenSaverProxy.get(), "Inhibit", parameters, G_DBUS_CALL_FLAGS_NONE, -1, m_cancellable.get(), [](GObject* proxy, GAsyncResult* result, gpointer userData) {
        GUniqueOutPtr<GError> error;
        GRefPtr<GVariant> returnValue = adoptGRef(g_dbus_proxy_call_finish(G_DBUS_PROXY(proxy), result, &error.outPtr()));
        if (g_error_matches(error.get(), G_IO_ERROR, G_IO_ERROR_CANCELLED))
            return;

        auto* self = static_cast<SleepDisablerGLib*>(userData);
        if (error)
            g_warning("Calling %s.Inhibit failed: %s", g_dbus_proxy_get_interface_name(G_DBUS_PROXY(proxy)), error->message);
        else {
            ASSERT(returnValue);
            if (shouldUsePortal())
                g_variant_get(returnValue.get(), "(o)", &self->m_inhibitPortalRequestObjectPath.outPtr());
            else
                g_variant_get(returnValue.get(), "(u)", &self->m_screenSaverCookie);
        }
        self->m_cancellable = nullptr;
    }, this);
}

void SleepDisablerGLib::releaseInhibitor()
{
    if (!shouldUsePortal()) {
        ASSERT(m_screenSaverCookie);
        g_dbus_proxy_call(m_screenSaverProxy.get(), "UnInhibit", g_variant_new("(u)", m_screenSaverCookie), G_DBUS_CALL_FLAGS_NONE, -1, nullptr, [](GObject* proxy, GAsyncResult* result, gpointer) {
            GUniqueOutPtr<GError> error;
            GRefPtr<GVariant> returnValue = adoptGRef(g_dbus_proxy_call_finish(G_DBUS_PROXY(proxy), result, &error.outPtr()));
            if (error)
                g_warning("Calling %s.UnInhibit failed: %s", g_dbus_proxy_get_interface_name(G_DBUS_PROXY(proxy)), error->message);
        }, nullptr);

        return;
    }

    ASSERT(m_inhibitPortalRequestObjectPath);
    g_dbus_proxy_new_for_bus(G_BUS_TYPE_SESSION, static_cast<GDBusProxyFlags>(G_DBUS_PROXY_FLAGS_DO_NOT_LOAD_PROPERTIES | G_DBUS_PROXY_FLAGS_DO_NOT_CONNECT_SIGNALS),
        nullptr, "org.freedesktop.portal.Desktop", m_inhibitPortalRequestObjectPath.get(), "org.freedesktop.portal.Request", nullptr, [](GObject*, GAsyncResult* result, gpointer) {
        GUniqueOutPtr<GError> error;
        GRefPtr<GDBusProxy> requestProxy = adoptGRef(g_dbus_proxy_new_for_bus_finish(result, &error.outPtr()));
        if (error) {
            g_warning("Failed to create org.freedesktop.portal.Request proxy: %s", error->message);
            return;
        }

        ASSERT(requestProxy);
        g_dbus_proxy_call(requestProxy.get(), "Close", g_variant_new("()"), G_DBUS_CALL_FLAGS_NONE, -1, nullptr, [](GObject* proxy, GAsyncResult* result, gpointer) {
            GUniqueOutPtr<GError> error;
            GRefPtr<GVariant> returnValue = adoptGRef(g_dbus_proxy_call_finish(G_DBUS_PROXY(proxy), result, &error.outPtr()));
            if (error)
                g_warning("Calling %s.Close failed: %s", g_dbus_proxy_get_interface_name(G_DBUS_PROXY(proxy)), error->message);
        }, nullptr);
    }, nullptr);
}

} // namespace PAL
