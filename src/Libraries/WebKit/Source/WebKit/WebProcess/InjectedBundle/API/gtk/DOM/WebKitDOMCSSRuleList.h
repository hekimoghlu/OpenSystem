/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

#ifndef WebKitDOMCSSRuleList_h
#define WebKitDOMCSSRuleList_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_CSS_RULE_LIST            (webkit_dom_css_rule_list_get_type())
#define WEBKIT_DOM_CSS_RULE_LIST(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_CSS_RULE_LIST, WebKitDOMCSSRuleList))
#define WEBKIT_DOM_CSS_RULE_LIST_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_CSS_RULE_LIST, WebKitDOMCSSRuleListClass)
#define WEBKIT_DOM_IS_CSS_RULE_LIST(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_CSS_RULE_LIST))
#define WEBKIT_DOM_IS_CSS_RULE_LIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_CSS_RULE_LIST))
#define WEBKIT_DOM_CSS_RULE_LIST_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_CSS_RULE_LIST, WebKitDOMCSSRuleListClass))

struct _WebKitDOMCSSRuleList {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMCSSRuleListClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_css_rule_list_get_type(void);

/**
 * webkit_dom_css_rule_list_item:
 * @self: A #WebKitDOMCSSRuleList
 * @index: A #gulong
 *
 * Returns: (transfer full): A #WebKitDOMCSSRule
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMCSSRule*
webkit_dom_css_rule_list_item(WebKitDOMCSSRuleList* self, gulong index);

/**
 * webkit_dom_css_rule_list_get_length:
 * @self: A #WebKitDOMCSSRuleList
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_css_rule_list_get_length(WebKitDOMCSSRuleList* self);

G_END_DECLS

#endif /* WebKitDOMCSSRuleList_h */
