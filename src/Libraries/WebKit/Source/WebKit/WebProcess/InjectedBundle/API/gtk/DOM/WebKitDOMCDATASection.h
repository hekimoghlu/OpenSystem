/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#ifndef WebKitDOMCDATASection_h
#define WebKitDOMCDATASection_h

#include <glib-object.h>
#include <webkitdom/WebKitDOMText.h>
#include <webkitdom/webkitdomdefines.h>

G_BEGIN_DECLS

#define WEBKIT_DOM_TYPE_CDATA_SECTION            (webkit_dom_cdata_section_get_type())
#define WEBKIT_DOM_CDATA_SECTION(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_DOM_TYPE_CDATA_SECTION, WebKitDOMCDATASection))
#define WEBKIT_DOM_CDATA_SECTION_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_DOM_TYPE_CDATA_SECTION, WebKitDOMCDATASectionClass)
#define WEBKIT_DOM_IS_CDATA_SECTION(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_DOM_TYPE_CDATA_SECTION))
#define WEBKIT_DOM_IS_CDATA_SECTION_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_DOM_TYPE_CDATA_SECTION))
#define WEBKIT_DOM_CDATA_SECTION_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_DOM_TYPE_CDATA_SECTION, WebKitDOMCDATASectionClass))

struct _WebKitDOMCDATASection {
    WebKitDOMText parent_instance;
};

struct _WebKitDOMCDATASectionClass {
    WebKitDOMTextClass parent_class;
};

WEBKIT_DEPRECATED GType
webkit_dom_cdata_section_get_type(void);

G_END_DECLS

#endif /* WebKitDOMCDATASection_h */
