/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef WebKitDOMXPathNSResolver_h
#define WebKitDOMXPathNSResolver_h

#include <glib-object.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_XPATH_NS_RESOLVER            (webkit_dom_xpath_ns_resolver_get_type ())
#define WEBKIT_DOM_XPATH_NS_RESOLVER(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_DOM_TYPE_XPATH_NS_RESOLVER, WebKitDOMXPathNSResolver))
#define WEBKIT_DOM_XPATH_NS_RESOLVER_CLASS(obj)      (G_TYPE_CHECK_CLASS_CAST ((obj), WEBKIT_DOM_TYPE_XPATH_NS_RESOLVER, WebKitDOMXPathNSResolverIface))
#define WEBKIT_DOM_IS_XPATH_NS_RESOLVER(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_DOM_TYPE_XPATH_NS_RESOLVER))
#define WEBKIT_DOM_XPATH_NS_RESOLVER_GET_IFACE(obj)  (G_TYPE_INSTANCE_GET_INTERFACE ((obj), WEBKIT_DOM_TYPE_XPATH_NS_RESOLVER, WebKitDOMXPathNSResolverIface))

struct _WebKitDOMXPathNSResolverIface {
    GTypeInterface gIface;

    /* virtual table */
    gchar *(* lookup_namespace_uri)(WebKitDOMXPathNSResolver *resolver,
                                    const gchar              *prefix);

    void (*_webkitdom_reserved0) (void);
    void (*_webkitdom_reserved1) (void);
    void (*_webkitdom_reserved2) (void);
    void (*_webkitdom_reserved3) (void);
};


WEBKIT_DEPRECATED GType webkit_dom_xpath_ns_resolver_get_type(void) G_GNUC_CONST;

/**
 * webkit_dom_xpath_ns_resolver_lookup_namespace_uri:
 * @resolver: A #WebKitDOMXPathNSResolver
 * @prefix: The prefix to lookup
 *
 * Returns: (transfer full): a #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gchar *webkit_dom_xpath_ns_resolver_lookup_namespace_uri(WebKitDOMXPathNSResolver *resolver,
                                                                    const gchar              *prefix);

G_END_DECLS

#endif /* WebKitDOMXPathNSResolver_h */
