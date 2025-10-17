/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#ifndef WebKitDOMAttr_h
#define WebKitDOMAttr_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMNode.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_ATTR            (webkit_dom_attr_get_type())
#define WEBKIT_DOM_ATTR(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_ATTR, WebKitDOMAttr))
#define WEBKIT_DOM_ATTR_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_ATTR, WebKitDOMAttrClass)
#define WEBKIT_DOM_IS_ATTR(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_ATTR))
#define WEBKIT_DOM_IS_ATTR_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_ATTR))
#define WEBKIT_DOM_ATTR_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_ATTR, WebKitDOMAttrClass))

struct _WebKitDOMAttr {
    WebKitDOMNode parent_instance;
};

struct _WebKitDOMAttrClass {
    WebKitDOMNodeClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_attr_get_type(void);

/**
 * webkit_dom_attr_get_name:
 * @self: A #WebKitDOMAttr
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_attr_get_name(WebKitDOMAttr* self);

/**
 * webkit_dom_attr_get_specified:
 * @self: A #WebKitDOMAttr
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_attr_get_specified(WebKitDOMAttr* self);

/**
 * webkit_dom_attr_get_value:
 * @self: A #WebKitDOMAttr
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_attr_get_value(WebKitDOMAttr* self);

/**
 * webkit_dom_attr_set_value:
 * @self: A #WebKitDOMAttr
 * @value: A #gchar
 * @error: #GError
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_attr_set_value(WebKitDOMAttr* self, const gchar* value, GError** error);

/**
 * webkit_dom_attr_get_owner_element:
 * @self: A #WebKitDOMAttr
 *
 * Returns: (transfer none): A #WebKitDOMElement
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMElement*
webkit_dom_attr_get_owner_element(WebKitDOMAttr* self);

/**
 * webkit_dom_attr_get_namespace_uri:
 * @self: A #WebKitDOMAttr
 *
 * Returns: A #gchar
 *
 * Since: 2.14
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_attr_get_namespace_uri(WebKitDOMAttr* self);

/**
 * webkit_dom_attr_get_prefix:
 * @self: A #WebKitDOMAttr
 *
 * Returns: A #gchar
 *
 * Since: 2.14
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_attr_get_prefix(WebKitDOMAttr* self);

/**
 * webkit_dom_attr_get_local_name:
 * @self: A #WebKitDOMAttr
 *
 * Returns: A #gchar
 *
 * Since: 2.14
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_attr_get_local_name(WebKitDOMAttr* self);

G_END_DECLS

#endif /* WebKitDOMAttr_h */
