/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#if !defined(__WEBKITDOM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <webkitdom/webkitdom.h> can be included directly."
#endif

#ifndef WebKitDOMHTMLOptionsCollection_h
#define WebKitDOMHTMLOptionsCollection_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLCollection.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_OPTIONS_COLLECTION            (webkit_dom_html_options_collection_get_type())
#define WEBKIT_DOM_HTML_OPTIONS_COLLECTION(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_OPTIONS_COLLECTION, WebKitDOMHTMLOptionsCollection))
#define WEBKIT_DOM_HTML_OPTIONS_COLLECTION_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_OPTIONS_COLLECTION, WebKitDOMHTMLOptionsCollectionClass)
#define WEBKIT_DOM_IS_HTML_OPTIONS_COLLECTION(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_OPTIONS_COLLECTION))
#define WEBKIT_DOM_IS_HTML_OPTIONS_COLLECTION_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_OPTIONS_COLLECTION))
#define WEBKIT_DOM_HTML_OPTIONS_COLLECTION_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_OPTIONS_COLLECTION, WebKitDOMHTMLOptionsCollectionClass))

struct _WebKitDOMHTMLOptionsCollection {
    WebKitDOMHTMLCollection parent_instance;
};

struct _WebKitDOMHTMLOptionsCollectionClass {
    WebKitDOMHTMLCollectionClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_options_collection_get_type(void);

/**
 * webkit_dom_html_options_collection_named_item:
 * @self: A #WebKitDOMHTMLOptionsCollection
 * @name: A #gchar
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_html_options_collection_named_item(WebKitDOMHTMLOptionsCollection* self, const gchar* name);

/**
 * webkit_dom_html_options_collection_get_selected_index:
 * @self: A #WebKitDOMHTMLOptionsCollection
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_options_collection_get_selected_index(WebKitDOMHTMLOptionsCollection* self);

/**
 * webkit_dom_html_options_collection_set_selected_index:
 * @self: A #WebKitDOMHTMLOptionsCollection
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_options_collection_set_selected_index(WebKitDOMHTMLOptionsCollection* self, glong value);

/**
 * webkit_dom_html_options_collection_get_length:
 * @self: A #WebKitDOMHTMLOptionsCollection
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_html_options_collection_get_length(WebKitDOMHTMLOptionsCollection* self);

G_END_DECLS

#endif /* WebKitDOMHTMLOptionsCollection_h */
