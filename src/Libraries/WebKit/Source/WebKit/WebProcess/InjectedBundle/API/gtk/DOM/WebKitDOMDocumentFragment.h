/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#ifndef WebKitDOMDocumentFragment_h
#define WebKitDOMDocumentFragment_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMNode.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_DOCUMENT_FRAGMENT            (webkit_dom_document_fragment_get_type())
#define WEBKIT_DOM_DOCUMENT_FRAGMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_DOCUMENT_FRAGMENT, WebKitDOMDocumentFragment))
#define WEBKIT_DOM_DOCUMENT_FRAGMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_DOCUMENT_FRAGMENT, WebKitDOMDocumentFragmentClass)
#define WEBKIT_DOM_IS_DOCUMENT_FRAGMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_DOCUMENT_FRAGMENT))
#define WEBKIT_DOM_IS_DOCUMENT_FRAGMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_DOCUMENT_FRAGMENT))
#define WEBKIT_DOM_DOCUMENT_FRAGMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_DOCUMENT_FRAGMENT, WebKitDOMDocumentFragmentClass))

struct _WebKitDOMDocumentFragment {
    WebKitDOMNode parent_instance;
};

struct _WebKitDOMDocumentFragmentClass {
    WebKitDOMNodeClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_document_fragment_get_type(void);

/**
 * webkit_dom_document_fragment_get_element_by_id:
 * @self: A #WebKitDOMDocumentFragment
 * @elementId: A #gchar
 *
 * Returns: (transfer none): A #WebKitDOMElement
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMElement*
webkit_dom_document_fragment_get_element_by_id(WebKitDOMDocumentFragment* self, const gchar* elementId);

/**
 * webkit_dom_document_fragment_query_selector:
 * @self: A #WebKitDOMDocumentFragment
 * @selectors: A #gchar
 * @error: #GError
 *
 * Returns: (transfer none): A #WebKitDOMElement
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMElement*
webkit_dom_document_fragment_query_selector(WebKitDOMDocumentFragment* self, const gchar* selectors, GError** error);

/**
 * webkit_dom_document_fragment_query_selector_all:
 * @self: A #WebKitDOMDocumentFragment
 * @selectors: A #gchar
 * @error: #GError
 *
 * Returns: (transfer full): A #WebKitDOMNodeList
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMNodeList*
webkit_dom_document_fragment_query_selector_all(WebKitDOMDocumentFragment* self, const gchar* selectors, GError** error);

/**
 * webkit_dom_document_fragment_get_children:
 * @self: A #WebKitDOMDocumentFragment
 *
 * Returns: (transfer full): A #WebKitDOMHTMLCollection
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMHTMLCollection*
webkit_dom_document_fragment_get_children(WebKitDOMDocumentFragment* self);

/**
 * webkit_dom_document_fragment_get_first_element_child:
 * @self: A #WebKitDOMDocumentFragment
 *
 * Returns: (transfer none): A #WebKitDOMElement
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMElement*
webkit_dom_document_fragment_get_first_element_child(WebKitDOMDocumentFragment* self);

/**
 * webkit_dom_document_fragment_get_last_element_child:
 * @self: A #WebKitDOMDocumentFragment
 *
 * Returns: (transfer none): A #WebKitDOMElement
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED WebKitDOMElement*
webkit_dom_document_fragment_get_last_element_child(WebKitDOMDocumentFragment* self);

/**
 * webkit_dom_document_fragment_get_child_element_count:
 * @self: A #WebKitDOMDocumentFragment
 *
 * Returns: A #gulong
 *
 * Since: 2.16
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gulong
webkit_dom_document_fragment_get_child_element_count(WebKitDOMDocumentFragment* self);

G_END_DECLS

#endif /* WebKitDOMDocumentFragment_h */
