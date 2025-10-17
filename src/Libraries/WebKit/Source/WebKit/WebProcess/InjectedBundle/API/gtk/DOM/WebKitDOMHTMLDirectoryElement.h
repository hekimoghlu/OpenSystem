/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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

#ifndef WebKitDOMHTMLDirectoryElement_h
#define WebKitDOMHTMLDirectoryElement_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMHTMLElement.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_HTML_DIRECTORY_ELEMENT            (webkit_dom_html_directory_element_get_type())
#define WEBKIT_DOM_HTML_DIRECTORY_ELEMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_HTML_DIRECTORY_ELEMENT, WebKitDOMHTMLDirectoryElement))
#define WEBKIT_DOM_HTML_DIRECTORY_ELEMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_HTML_DIRECTORY_ELEMENT, WebKitDOMHTMLDirectoryElementClass)
#define WEBKIT_DOM_IS_HTML_DIRECTORY_ELEMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_HTML_DIRECTORY_ELEMENT))
#define WEBKIT_DOM_IS_HTML_DIRECTORY_ELEMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_HTML_DIRECTORY_ELEMENT))
#define WEBKIT_DOM_HTML_DIRECTORY_ELEMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_HTML_DIRECTORY_ELEMENT, WebKitDOMHTMLDirectoryElementClass))

struct _WebKitDOMHTMLDirectoryElement {
    WebKitDOMHTMLElement parent_instance;
};

struct _WebKitDOMHTMLDirectoryElementClass {
    WebKitDOMHTMLElementClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_html_directory_element_get_type(void);

/**
 * webkit_dom_html_directory_element_get_compact:
 * @self: A #WebKitDOMHTMLDirectoryElement
 *
 * Returns: A #gboolean
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED gboolean
webkit_dom_html_directory_element_get_compact(WebKitDOMHTMLDirectoryElement* self);

/**
 * webkit_dom_html_directory_element_set_compact:
 * @self: A #WebKitDOMHTMLDirectoryElement
 * @value: A #gboolean
 *
 *
 * Deprecated: 2.22: Use JavaScriptCore API instead
**/
WEBKIT_DEPRECATED void
webkit_dom_html_directory_element_set_compact(WebKitDOMHTMLDirectoryElement* self, gboolean value);

G_END_DECLS

#endif /* WebKitDOMHTMLDirectoryElement_h */
