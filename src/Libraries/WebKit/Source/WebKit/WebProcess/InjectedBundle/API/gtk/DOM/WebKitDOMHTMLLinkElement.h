/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#ifndef WebKitDOMHTMLLinkElement_h
#define WebKitDOMHTMLLinkElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_LINK_ELEMENT            (webkit_dom_html_link_element_get_type())
#define WEBKIT_DOM_HTML_LINK_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_LINK_ELEMENT, WebKitDOMHTMLLinkElement))
#define WEBKIT_DOM_HTML_LINK_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_LINK_ELEMENT, WebKitDOMHTMLLinkElementClass)
#define WEBKIT_DOM_IS_HTML_LINK_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_LINK_ELEMENT))
#define WEBKIT_DOM_IS_HTML_LINK_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_LINK_ELEMENT))
#define WEBKIT_DOM_HTML_LINK_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_LINK_ELEMENT, WebKitDOMHTMLLinkElementClass))

struct _WebKitDOMHTMLLinkElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLLinkElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_link_element_get_type(void);

/**
 * webkit_dom_html_link_element_get_disabled:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_link_element_get_disabled(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_disabled:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_disabled(WebKitDOMHTMLLinkElement* self, gboolean value);

/**
 * webkit_dom_html_link_element_get_charset:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_charset(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_charset:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_charset(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_href:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_href(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_href:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_href(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_hreflang:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_hreflang(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_hreflang:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_hreflang(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_media:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_media(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_media:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_media(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_rel:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_rel(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_rel:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_rel(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_rev:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_rev(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_rev:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_rev(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_target:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_target(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_target:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_target(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_type_attr:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_link_element_get_type_attr(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_type_attr:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_type_attr(WebKitDOMHTMLLinkElement* self, const gchar* value);

/**
 * webkit_dom_html_link_element_get_sheet:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: (transfer full): A #WebKitDOMStyleSheet
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMStyleSheet*
webkit_dom_html_link_element_get_sheet(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_get_sizes:
 * @self: A #WebKitDOMHTMLLinkElement
 *
 * Returns: (transfer full): A #WebKitDOMDOMTokenList
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMDOMTokenList*
webkit_dom_html_link_element_get_sizes(WebKitDOMHTMLLinkElement* self);

/**
 * webkit_dom_html_link_element_set_sizes:
 * @self: A #WebKitDOMHTMLLinkElement
 * @value: a #gchar
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_link_element_set_sizes(WebKitDOMHTMLLinkElement* self, const gchar* value);

G_END_DECLS

#endif /* WebKitDOMHTMLLinkElement_h */
