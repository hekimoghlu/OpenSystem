/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
#error "Only <wpe/webkitdom.h> can be included directly."
#endif

#ifndef WebKitDOMNode_h
#define WebKitDOMNode_h

#include <glib-object.h>
#include <jsc/jsc.h>
#include <wpe/WebKitDOMObject.h>
#include <wpe/WebKitDOMDefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_NODE            (webkit_dom_node_get_type())
#define WEBKIT_DOM_NODE(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_NODE, WebKitDOMNode))
#define WEBKIT_DOM_NODE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_NODE, WebKitDOMNodeClass)
#define WEBKIT_DOM_IS_NODE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_NODE))
#define WEBKIT_DOM_IS_NODE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_NODE))
#define WEBKIT_DOM_NODE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_NODE, WebKitDOMNodeClass))

typedef struct _WebKitDOMNode WebKitDOMNode;
typedef struct _WebKitDOMNodeClass WebKitDOMNodeClass;
typedef struct _WebKitDOMNodePrivate WebKitDOMNodePrivate;

struct _WebKitDOMNode {
    WebKitDOMObject parent_instance;

    WebKitDOMNodePrivate *priv;
};

struct _WebKitDOMNodeClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_node_get_type     (void);

WEBKIT_DEPRECATED WebKitDOMNode *
webkit_dom_node_for_js_value (JSCValue *value);

G_END_DECLS

#endif /* WebKitDOMNode_h */
