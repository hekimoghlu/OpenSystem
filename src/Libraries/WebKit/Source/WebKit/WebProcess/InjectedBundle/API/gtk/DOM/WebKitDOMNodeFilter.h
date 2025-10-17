/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#ifndef WebKitDOMNodeFilter_h
#define WebKitDOMNodeFilter_h

#include <glib-object.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_NODE_FILTER            (webkit_dom_node_filter_get_type ())
#define WEBKIT_DOM_NODE_FILTER(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_DOM_TYPE_NODE_FILTER, WebKitDOMNodeFilter))
#define WEBKIT_DOM_NODE_FILTER_CLASS(obj)      (G_TYPE_CHECK_CLASS_CAST ((obj), WEBKIT_DOM_TYPE_NODE_FILTER, WebKitDOMNodeFilterIface))
#define WEBKIT_DOM_IS_NODE_FILTER(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_DOM_TYPE_NODE_FILTER))
#define WEBKIT_DOM_NODE_FILTER_GET_IFACE(obj)  (G_TYPE_INSTANCE_GET_INTERFACE ((obj), WEBKIT_DOM_TYPE_NODE_FILTER, WebKitDOMNodeFilterIface))

#ifndef WEBKIT_DISABLE_DEPRECATED

/**
 * WEBKIT_DOM_NODE_FILTER_ACCEPT:
 *
 * Accept the node. Use this macro as return value of webkit_dom_node_filter_accept_node()
 * implementation to accept the given #WebKitDOMNode
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_ACCEPT 1

/**
 * WEBKIT_DOM_NODE_FILTER_REJECT:
 *
 * Reject the node. Use this macro as return value of webkit_dom_node_filter_accept_node()
 * implementation to reject the given #WebKitDOMNode. The children of the given node will
 * be rejected too.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_REJECT 2

/**
 * WEBKIT_DOM_NODE_FILTER_SKIP:
 *
 * Skip the node. Use this macro as return value of webkit_dom_node_filter_accept_node()
 * implementation to skip the given #WebKitDOMNode. The children of the given node will
 * not be skipped.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SKIP   3

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_ALL:
 *
 * Show all nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_ALL                    0xFFFFFFFF

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_ELEMENT:
 *
 * Show #WebKitDOMElement nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_ELEMENT                0x00000001

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_ATTRIBUTE:
 *
 * Show #WebKitDOMAttr nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_ATTRIBUTE              0x00000002

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_TEXT:
 *
 * Show #WebKitDOMText nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_TEXT                   0x00000004

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_CDATA_SECTION:
 *
 * Show #WebKitDOMCDataSection nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_CDATA_SECTION          0x00000008

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_ENTITY_REFERENCE:
 *
 * Show #WebKitDOMEntityReference nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_ENTITY_REFERENCE       0x00000010

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_ENTITY:
 *
 * Show #WebKitDOMEntity nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_ENTITY                 0x00000020

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_PROCESSING_INSTRUCTION:
 *
 * Show #WebKitDOMProcessingInstruction nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_PROCESSING_INSTRUCTION 0x00000040

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_COMMENT:
 *
 * Show #WebKitDOMComment nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_COMMENT                0x00000080

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_DOCUMENT:
 *
 * Show #WebKitDOMDocument nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_DOCUMENT               0x00000100

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_DOCUMENT_TYPE:
 *
 * Show #WebKitDOMDocumentType nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_DOCUMENT_TYPE          0x00000200

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_DOCUMENT_FRAGMENT:
 *
 * Show #WebKitDOMDocumentFragment nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_DOCUMENT_FRAGMENT      0x00000400

/**
 * WEBKIT_DOM_NODE_FILTER_SHOW_NOTATION:
 *
 * Show #WebKitDOMNotation nodes.
 *
 * Since: 2.6
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
#define WEBKIT_DOM_NODE_FILTER_SHOW_NOTATION               0x00000800

#endif /* WEBKIT_DISABLE_DEPRECATED */

struct _WebKitDOMNodeFilterIface {
    GTypeInterface gIface;

    /* virtual table */
    gshort (* accept_node)(WebKitDOMNodeFilter *filter,
                           WebKitDOMNode       *node);

    void (*_webkitdom_reserved0) (void);
    void (*_webkitdom_reserved1) (void);
    void (*_webkitdom_reserved2) (void);
    void (*_webkitdom_reserved3) (void);
};


WEBKIT_DEPRECATED GType webkit_dom_node_filter_get_type(void) G_GNUC_CONST;

/**
 * webkit_dom_node_filter_accept_node:
 * @filter: A #WebKitDOMNodeFilter
 * @node: A #WebKitDOMNode
 *
 * Returns: a #gshort
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
 */
WEBKIT_DEPRECATED gshort webkit_dom_node_filter_accept_node(WebKitDOMNodeFilter *filter,
                                                     WebKitDOMNode       *node);

G_END_DECLS

#endif /* WebKitDOMNodeFilter_h */
