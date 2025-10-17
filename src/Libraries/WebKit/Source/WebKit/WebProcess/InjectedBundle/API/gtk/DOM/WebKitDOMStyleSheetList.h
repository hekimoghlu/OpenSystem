/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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

#ifndef WebKitDOMStyleSheetList_h
#define WebKitDOMStyleSheetList_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMObject.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_STYLE_SHEET_LIST            (webkit_dom_style_sheet_list_get_type())
#define WEBKIT_DOM_STYLE_SHEET_LIST(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_STYLE_SHEET_LIST, WebKitDOMStyleSheetList))
#define WEBKIT_DOM_STYLE_SHEET_LIST_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_STYLE_SHEET_LIST, WebKitDOMStyleSheetListClass)
#define WEBKIT_DOM_IS_STYLE_SHEET_LIST(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_STYLE_SHEET_LIST))
#define WEBKIT_DOM_IS_STYLE_SHEET_LIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_STYLE_SHEET_LIST))
#define WEBKIT_DOM_STYLE_SHEET_LIST_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_STYLE_SHEET_LIST, WebKitDOMStyleSheetListClass))

struct _WebKitDOMStyleSheetList {
    WebKitDOMObject parent_instance;
};

struct _WebKitDOMStyleSheetListClass {
    WebKitDOMObjectClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_style_sheet_list_get_type(void);

/**
 * webkit_dom_style_sheet_list_item:
 * @self: A #WebKitDOMStyleSheetList
 * @index: A #gulong
 *
 * Returns: (transfer full): A #WebKitDOMStyleSheet
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMStyleSheet*
webkit_dom_style_sheet_list_item(WebKitDOMStyleSheetList* self, gulong index);

/**
 * webkit_dom_style_sheet_list_get_length:
 * @self: A #WebKitDOMStyleSheetList
 *
 * Returns: A #gulong
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gulong
webkit_dom_style_sheet_list_get_length(WebKitDOMStyleSheetList* self);

G_END_DECLS

#endif /* WebKitDOMStyleSheetList_h */
