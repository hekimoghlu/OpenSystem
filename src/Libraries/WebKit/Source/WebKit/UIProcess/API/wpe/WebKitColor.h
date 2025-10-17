/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#if !defined(__WEBKIT_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/webkit.h> can be included directly."
#endif

#ifndef WebKitColor_h
#define WebKitColor_h

#include <glib-object.h>
#include <wpe/WebKitDefines.h>

G_BEGIN_DECLS

struct _WebKitColor {
    gdouble red;
    gdouble green;
    gdouble blue;
    gdouble alpha;
};

typedef struct _WebKitColor WebKitColor;

#define WEBKIT_TYPE_COLOR (webkit_color_get_type())

WEBKIT_API GType
webkit_color_get_type     (void);

WEBKIT_API gboolean
webkit_color_parse        (WebKitColor *color, const gchar *color_string);

G_END_DECLS

#endif /* WebKitColor_h */
