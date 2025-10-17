/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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

#ifndef WebKitDOMText_h
#define WebKitDOMText_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMCharacterData.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_TEXT            (webkit_dom_text_get_type())
#define WEBKIT_DOM_TEXT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_TEXT, WebKitDOMText))
#define WEBKIT_DOM_TEXT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_TEXT, WebKitDOMTextClass)
#define WEBKIT_DOM_IS_TEXT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_TEXT))
#define WEBKIT_DOM_IS_TEXT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_TEXT))
#define WEBKIT_DOM_TEXT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_TEXT, WebKitDOMTextClass))

struct _WebKitDOMText {
    WebKitDOMCharacterData parent_instance;
};

struct _WebKitDOMTextClass {
    WebKitDOMCharacterDataClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_text_get_type(void);

/**
 * webkit_dom_text_split_text:
 * @self: A #WebKitDOMText
 * @offset: A #gulong
 * @error: #GError
 *
 * Returns: (transfer none): A #WebKitDOMText
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED WebKitDOMText*
webkit_dom_text_split_text(WebKitDOMText* self, gulong offset, GError** error);

/**
 * webkit_dom_text_get_whole_text:
 * @self: A #WebKitDOMText
 *
 * Returns: A #gchar
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gchar*
webkit_dom_text_get_whole_text(WebKitDOMText* self);

G_END_DECLS

#endif /* WebKitDOMText_h */
