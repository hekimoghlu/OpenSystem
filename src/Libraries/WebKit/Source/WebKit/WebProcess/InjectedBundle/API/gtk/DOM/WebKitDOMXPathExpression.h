/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

#ifndef WebKitDOMXPathExpression_h
#define WebKitDOMXPathExpression_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_XPATH_EXPRESSION            (webkit_dom_xpath_expression_get_type())
#define WEBKIT_DOM_XPATH_EXPRESSION(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_XPATH_EXPRESSION, WebKitDOMXPathExpression))
#define WEBKIT_DOM_XPATH_EXPRESSION_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_XPATH_EXPRESSION, WebKitDOMXPathExpressionClass)
#define WEBKIT_DOM_IS_XPATH_EXPRESSION(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_XPATH_EXPRESSION))
#define WEBKIT_DOM_IS_XPATH_EXPRESSION_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_XPATH_EXPRESSION))
#define WEBKIT_DOM_XPATH_EXPRESSION_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_XPATH_EXPRESSION, WebKitDOMXPathExpressionClass))

struct _WebKitDOMXPathExpression {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMXPathExpressionClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_xpath_expression_get_type(void);

/**
 * webkit_dom_xpath_expression_evaluate:
 * @self: A #WebKitDOMXPathExpression
 * @contextNode: A #WebKitDOMNode
 * @type: A #gushort
 * @inResult: A #WebKitDOMXPathResult
 * @error: #GError
 *
 * Returns: (transfer full): A #WebKitDOMXPathResult
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMXPathResult*
webkit_dom_xpath_expression_evaluate(WebKitDOMXPathExpression* self, WebKitDOMNode* contextNode, gushort type, WebKitDOMXPathResult* inResult, GError** error);

G_END_DECLS

#endif /* WebKitDOMXPathExpression_h */
