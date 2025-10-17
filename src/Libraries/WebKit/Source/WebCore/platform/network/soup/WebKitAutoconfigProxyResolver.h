/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#pragma once

#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

#define WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER            (webkit_autoconfig_proxy_resolver_get_type ())
#define WEBKIT_AUTOCONFIG_PROXY_RESOLVER(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER, WebKitAutoconfigProxyResolver))
#define WEBKIT_IS_AUTOCONFIG_PROXY_RESOLVER(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER))
#define WEBKIT_AUTOCONFIG_PROXY_RESOLVER_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER, WebKitAutoconfigProxyResolverClass))
#define WEBKIT_IS_AUTOCONFIG_PROXY_RESOLVER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((obj), WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER))
#define WEBKIT_AUTOCONFIG_PROXY_RESOLVER_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), WEBKIT_TYPE_AUTOCONFIG_PROXY_RESOLVER, WebKitAutoconfigProxyResolverClass))

typedef struct _WebKitAutoconfigProxyResolver WebKitAutoconfigProxyResolver;
typedef struct _WebKitAutoconfigProxyResolverClass WebKitAutoconfigProxyResolverClass;
typedef struct _WebKitAutoconfigProxyResolverPrivate WebKitAutoconfigProxyResolverPrivate;

struct _WebKitAutoconfigProxyResolver {
    GObject parent;

    WebKitAutoconfigProxyResolverPrivate* priv;
};

struct _WebKitAutoconfigProxyResolverClass {
    GObjectClass parentClass;
};

GType webkit_autoconfig_proxy_resolver_get_type(void);

GRefPtr<GProxyResolver> webkitAutoconfigProxyResolverNew(const CString&);
