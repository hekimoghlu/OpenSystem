/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#include "WebKitAutoconfigProxyResolver.h"

#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/WTFGType.h>

struct _WebKitAutoconfigProxyResolverPrivate {
    GRefPtr<GDBusProxy> pacRunner;
    CString autoconfigURL;
};

static void webkitAutoconfigProxyResolverInterfaceInit(GProxyResolverInterface*);

WEBKIT_DEFINE_TYPE_WITH_CODE(WebKitAutoconfigProxyResolver, webkit_autoconfig_proxy_resolver, G_TYPE_OBJECT,
    G_IMPLEMENT_INTERFACE(G_TYPE_PROXY_RESOLVER, webkitAutoconfigProxyResolverInterfaceInit))

static void webkit_autoconfig_proxy_resolver_class_init(WebKitAutoconfigProxyResolverClass*)
{
}

GRefPtr<GProxyResolver> webkitAutoconfigProxyResolverNew(const CString& autoconfigURL)
{
    GUniqueOutPtr<GError> error;
    GRefPtr<GDBusProxy> pacRunner = adoptGRef(g_dbus_proxy_new_for_bus_sync(G_BUS_TYPE_SESSION,
        static_cast<GDBusProxyFlags>(G_DBUS_PROXY_FLAGS_DO_NOT_LOAD_PROPERTIES | G_DBUS_PROXY_FLAGS_DO_NOT_CONNECT_SIGNALS),
        nullptr, "org.gtk.GLib.PACRunner", "/org/gtk/GLib/PACRunner", "org.gtk.GLib.PACRunner", nullptr, &error.outPtr()));
    if (!pacRunner) {
        g_warning("Could not start proxy autoconfiguration helper: %s\n", error->message);
        return nullptr;
    }

    auto* proxyResolver = WEBKIT_AUTOCONFIG_PROXY_RESOLVER(g_object_new(WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER, nullptr));
    proxyResolver->priv->pacRunner = WTFMove(pacRunner);
    proxyResolver->priv->autoconfigURL = autoconfigURL;

    return adoptGRef(G_PROXY_RESOLVER(proxyResolver));
}

static gchar** webkitAutoconfigProxyResolverLookup(GProxyResolver* proxyResolver, const char* uri, GCancellable* cancellable, GError** error)
{
    auto* priv = WEBKIT_AUTOCONFIG_PROXY_RESOLVER(proxyResolver)->priv;
    GRefPtr<GVariant> variant = adoptGRef(g_dbus_proxy_call_sync(priv->pacRunner.get(), "Lookup", g_variant_new("(ss)", priv->autoconfigURL.data(), uri),
        G_DBUS_CALL_FLAGS_NONE, -1, cancellable, error));
    if (!variant)
        return nullptr;

    gchar** proxies;
    g_variant_get(variant.get(), "(^as)", &proxies);
    return proxies;
}

static void webkitAutoconfigProxyResolverLookupAsync(GProxyResolver* proxyResolver, const char* uri, GCancellable* cancellable, GAsyncReadyCallback callback, gpointer userData)
{
    GTask* task = g_task_new(proxyResolver, cancellable, callback, userData);
    auto* priv = WEBKIT_AUTOCONFIG_PROXY_RESOLVER(proxyResolver)->priv;
    g_dbus_proxy_call(priv->pacRunner.get(), "Lookup", g_variant_new("(ss)", priv->autoconfigURL.data(), uri), G_DBUS_CALL_FLAGS_NONE, -1, cancellable,
        [](GObject* source, GAsyncResult* result, gpointer userData) {
            GRefPtr<GTask> task = adoptGRef(G_TASK(userData));
            GUniqueOutPtr<GError> error;
            GRefPtr<GVariant> variant = adoptGRef(g_dbus_proxy_call_finish(G_DBUS_PROXY(source), result, &error.outPtr()));
            if (variant) {
                gchar** proxies;
                g_variant_get(variant.get(), "(^as)", &proxies);
                g_task_return_pointer(task.get(), proxies, reinterpret_cast<GDestroyNotify>(g_strfreev));
            } else
                g_task_return_error(task.get(), error.release());
        }, task);
}

static gchar** webkitAutoconfigProxyResolverLookupFinish(GProxyResolver*, GAsyncResult* result, GError** error)
{
    return static_cast<char**>(g_task_propagate_pointer(G_TASK(result), error));
}

static void webkitAutoconfigProxyResolverInterfaceInit(GProxyResolverInterface* interface)
{
    interface->lookup = webkitAutoconfigProxyResolverLookup;
    interface->lookup_async = webkitAutoconfigProxyResolverLookupAsync;
    interface->lookup_finish = webkitAutoconfigProxyResolverLookupFinish;
}
