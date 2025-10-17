/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_CACHED_RESOLVER            (webkit_cached_resolver_get_type())
#define WEBKIT_CACHED_RESOLVER(object)         (G_TYPE_CHECK_INSTANCE_CAST((object), WEBKIT_TYPE_CACHED_RESOLVER, WebKitCachedResolver))
#define WEBKIT_IS_CACHED_RESOLVER(object)      (G_TYPE_CHECK_INSTANCE_TYPE((object), WEBKIT_TYPE_CACHED_RESOLVER))
#define WEBKIT_CACHED_RESOLVER_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_CACHED_RESOLVER, WebKitCachedResolverClass))
#define WEBKIT_IS_CACHED_RESOLVER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_CACHED_RESOLVER))
#define WEBKIT_CACHED_RESOLVER_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj), WEBKIT_TYPE_CACHED_RESOLVER, WebKitCachedResolverClass))

typedef struct _WebKitCachedResolver WebKitCachedResolver;
typedef struct _WebKitCachedResolverClass WebKitCachedResolverClass;

GType webkit_cached_resolver_get_type(void);

GResolver* webkitCachedResolverNew(GRefPtr<GResolver>&&);

G_END_DECLS
