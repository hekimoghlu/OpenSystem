/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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

#ifndef WebKitDOMHTMLMapElement_h
#define WebKitDOMHTMLMapElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_MAP_ELEMENT            (webkit_dom_html_map_element_get_type())
#define WEBKIT_DOM_HTML_MAP_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_MAP_ELEMENT, WebKitDOMHTMLMapElement))
#define WEBKIT_DOM_HTML_MAP_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_MAP_ELEMENT, WebKitDOMHTMLMapElementClass)
#define WEBKIT_DOM_IS_HTML_MAP_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_MAP_ELEMENT))
#define WEBKIT_DOM_IS_HTML_MAP_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_MAP_ELEMENT))
#define WEBKIT_DOM_HTML_MAP_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_MAP_ELEMENT, WebKitDOMHTMLMapElementClass))

struct _WebKitDOMHTMLMapElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLMapElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_map_element_get_type(void);

/**
 * webkit_dom_html_map_element_get_areas:
 * @self: A #WebKitDOMHTMLMapElement
 *
 * Returns: (transfer full): A #WebKitDOMHTMLCollection
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMHTMLCollection*
webkit_dom_html_map_element_get_areas(WebKitDOMHTMLMapElement* self);

/**
 * webkit_dom_html_map_element_get_name:
 * @self: A #WebKitDOMHTMLMapElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_map_element_get_name(WebKitDOMHTMLMapElement* self);

/**
 * webkit_dom_html_map_element_set_name:
 * @self: A #WebKitDOMHTMLMapElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_map_element_set_name(WebKitDOMHTMLMapElement* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLMapElement_h */
