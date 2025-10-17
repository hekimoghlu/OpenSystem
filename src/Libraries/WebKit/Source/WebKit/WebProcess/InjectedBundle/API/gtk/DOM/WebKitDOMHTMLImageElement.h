/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#ifndef WebKitDOMHTMLImageElement_h
#define WebKitDOMHTMLImageElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_IMAGE_ELEMENT            (webkit_dom_html_image_element_get_type())
#define WEBKIT_DOM_HTML_IMAGE_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_IMAGE_ELEMENT, WebKitDOMHTMLImageElement))
#define WEBKIT_DOM_HTML_IMAGE_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_IMAGE_ELEMENT, WebKitDOMHTMLImageElementClass)
#define WEBKIT_DOM_IS_HTML_IMAGE_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_IMAGE_ELEMENT))
#define WEBKIT_DOM_IS_HTML_IMAGE_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_IMAGE_ELEMENT))
#define WEBKIT_DOM_HTML_IMAGE_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_IMAGE_ELEMENT, WebKitDOMHTMLImageElementClass))

struct _WebKitDOMHTMLImageElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLImageElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_image_element_get_type(void);

/**
 * webkit_dom_html_image_element_get_name:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_name(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_name:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_name(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_align:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_align(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_align:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_align(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_alt:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_alt(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_alt:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_alt(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_border:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_border(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_border:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_border(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_height:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_height(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_height:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_height(WebKitDOMHTMLImageElement* self, glong value);

/**
 * webkit_dom_html_image_element_get_hspace:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_hspace(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_hspace:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_hspace(WebKitDOMHTMLImageElement* self, glong value);

/**
 * webkit_dom_html_image_element_get_is_map:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_image_element_get_is_map(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_is_map:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_is_map(WebKitDOMHTMLImageElement* self, gboolean value);

/**
 * webkit_dom_html_image_element_get_long_desc:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_long_desc(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_long_desc:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_long_desc(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_src:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_src(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_src:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_src(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_use_map:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_use_map(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_use_map:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_use_map(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_vspace:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_vspace(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_vspace:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_vspace(WebKitDOMHTMLImageElement* self, glong value);

/**
 * webkit_dom_html_image_element_get_width:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_width(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_width:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #glong
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_width(WebKitDOMHTMLImageElement* self, glong value);

/**
 * webkit_dom_html_image_element_get_complete:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_image_element_get_complete(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_get_lowsrc:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_html_image_element_get_lowsrc(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_set_lowsrc:
 * @self: A #WebKitDOMHTMLImageElement
 * @value: A #gchar
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_image_element_set_lowsrc(WebKitDOMHTMLImageElement* self, const gchar* value);

/**
 * webkit_dom_html_image_element_get_natural_height:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_natural_height(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_get_natural_width:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_natural_width(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_get_x:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_x(WebKitDOMHTMLImageElement* self);

/**
 * webkit_dom_html_image_element_get_y:
 * @self: A #WebKitDOMHTMLImageElement
 *
 * Returns: A #glong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED glong
webkit_dom_html_image_element_get_y(WebKitDOMHTMLImageElement* self);

G_END_DECLS

#endif /* WebKitDOMHTMLImageElement_h */
