/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

#ifndef WebKitDOMNodeIterator_h
#define WebKitDOMNodeIterator_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_NODE_ITERATOR            (webkit_dom_node_iterator_get_type())
#define WEBKIT_DOM_NODE_ITERATOR(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_NODE_ITERATOR, WebKitDOMNodeIterator))
#define WEBKIT_DOM_NODE_ITERATOR_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_NODE_ITERATOR, WebKitDOMNodeIteratorClass)
#define WEBKIT_DOM_IS_NODE_ITERATOR(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_NODE_ITERATOR))
#define WEBKIT_DOM_IS_NODE_ITERATOR_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_NODE_ITERATOR))
#define WEBKIT_DOM_NODE_ITERATOR_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_NODE_ITERATOR, WebKitDOMNodeIteratorClass))

struct _WebKitDOMNodeIterator {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMNodeIteratorClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_node_iterator_get_type(void);

/**
 * webkit_dom_node_iterator_next_node:
 * @self: A #WebKitDOMNodeIterator
 * @error: #GError
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_node_iterator_next_node(WebKitDOMNodeIterator* self, GError** error);

/**
 * webkit_dom_node_iterator_previous_node:
 * @self: A #WebKitDOMNodeIterator
 * @error: #GError
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_node_iterator_previous_node(WebKitDOMNodeIterator* self, GError** error);

/**
 * webkit_dom_node_iterator_detach:
 * @self: A #WebKitDOMNodeIterator
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_node_iterator_detach(WebKitDOMNodeIterator* self);

/**
 * webkit_dom_node_iterator_get_root:
 * @self: A #WebKitDOMNodeIterator
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_node_iterator_get_root(WebKitDOMNodeIterator* self);

/**
 * webkit_dom_node_iterator_get_what_to_show:
 * @self: A #WebKitDOMNodeIterator
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_node_iterator_get_what_to_show(WebKitDOMNodeIterator* self);

/**
 * webkit_dom_node_iterator_get_filter:
 * @self: A #WebKitDOMNodeIterator
 *
 * Returns: (transfer full): A #WebKitDOMNodeFilter
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMNodeFilter*
webkit_dom_node_iterator_get_filter(WebKitDOMNodeIterator* self);

/**
 * webkit_dom_node_iterator_get_reference_node:
 * @self: A #WebKitDOMNodeIterator
 *
 * Returns: (transfer none): A #WebKitDOMNode
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMNode*
webkit_dom_node_iterator_get_reference_node(WebKitDOMNodeIterator* self);

/**
 * webkit_dom_node_iterator_get_pointer_before_reference_node:
 * @self: A #WebKitDOMNodeIterator
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_node_iterator_get_pointer_before_reference_node(WebKitDOMNodeIterator* self);

G_END_DECLS

#endif /* WebKitDOMNodeIterator_h */
