/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#ifndef WebKitDOMDocument_h
#define WebKitDOMDocument_h

#include <glib-object.h>
#include <wpe/WebKitDOMNode.h>
#include <wpe/WebKitDOMDefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_DOCUMENT            (webkit_dom_document_get_type())
#define WEBKIT_DOM_DOCUMENT(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_DOCUMENT, WebKitDOMDocument))
#define WEBKIT_DOM_DOCUMENT_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_DOCUMENT, WebKitDOMDocumentClass)
#define WEBKIT_DOM_IS_DOCUMENT(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_DOCUMENT))
#define WEBKIT_DOM_IS_DOCUMENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_DOCUMENT))
#define WEBKIT_DOM_DOCUMENT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_DOCUMENT, WebKitDOMDocumentClass))

typedef struct _WebKitDOMDocument WebKitDOMDocument;
typedef struct _WebKitDOMDocumentClass WebKitDOMDocumentClass;

struct _WebKitDOMDocument {
    WebKitDOMNode parent_instance;
};

struct _WebKitDOMDocumentClass {
    WebKitDOMNodeClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_document_get_type(void);

G_END_DECLS

#endif /* WebKitDOMDocument_h */
