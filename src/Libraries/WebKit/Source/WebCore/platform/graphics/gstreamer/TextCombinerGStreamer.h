/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#pragma once

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include <gst/gst.h>

#define WEBKIT_TYPE_TEXT_COMBINER webkit_text_combiner_get_type()

#define WEBKIT_TEXT_COMBINER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_TEXT_COMBINER, WebKitTextCombiner))
#define WEBKIT_TEXT_COMBINER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_TEXT_COMBINER, WebKitTextCombinerClass))
#define WEBKIT_IS_TEXT_COMBINER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_TEXT_COMBINER))
#define WEBKIT_IS_TEXT_COMBINER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_TEXT_COMBINER))
#define WEBKIT_TEXT_COMBINER_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), WEBKIT_TYPE_TEXT_COMBINER, WebKitTextCombinerClass))

GType webkit_text_combiner_get_type(void);

typedef struct _WebKitTextCombiner WebKitTextCombiner;
typedef struct _WebKitTextCombinerClass WebKitTextCombinerClass;
typedef struct _WebKitTextCombinerPrivate WebKitTextCombinerPrivate;

struct _WebKitTextCombiner {
    GstBin parent;

    WebKitTextCombinerPrivate* priv;
};

struct _WebKitTextCombinerClass {
    GstBinClass parentClass;
};

GstElement* webkitTextCombinerNew();

void webKitTextCombinerHandleCaps(WebKitTextCombiner*, GstPad*, const GstCaps*);

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
