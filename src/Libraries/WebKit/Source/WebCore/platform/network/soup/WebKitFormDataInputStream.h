/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

#include "FormData.h"
#include <gio/gio.h>
#include <wtf/Forward.h>
#include <wtf/glib/GRefPtr.h>

#define WEBKIT_TYPE_FORM_DATA_INPUT_STREAM            (webkit_form_data_input_stream_get_type ())
#define WEBKIT_FORM_DATA_INPUT_STREAM(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_TYPE_FORM_DATA_INPUT_STREAM, WebKitFormDataInputStream))
#define WEBKIT_IS_FORM_DATA_INPUT_STREAM(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_TYPE_FORM_DATA_INPUT_STREAM))
#define WEBKIT_FORM_DATA_INPUT_STREAM_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), WEBKIT_TYPE_FORM_DATA_INPUT_STREAM, WebKitFormDataInputStreamClass))
#define WEBKIT_IS_FORM_DATA_INPUT_STREAM_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((obj), WEBKIT_TYPE_FORM_DATA_INPUT_STREAM))
#define WEBKIT_FORM_DATA_INPUT_STREAM_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), WEBKIT_TYPE_FORM_DATA_INPUT_STREAM, WebKitFormDataInputStreamClass))

typedef struct _WebKitFormDataInputStream WebKitFormDataInputStream;
typedef struct _WebKitFormDataInputStreamClass WebKitFormDataInputStreamClass;
typedef struct _WebKitFormDataInputStreamPrivate WebKitFormDataInputStreamPrivate;

struct _WebKitFormDataInputStream {
    GInputStream parent;

    WebKitFormDataInputStreamPrivate* priv;
};

struct _WebKitFormDataInputStreamClass {
    GInputStreamClass parentClass;
};

GType webkit_form_data_input_stream_get_type(void);

GRefPtr<GInputStream> webkitFormDataInputStreamNew(Ref<WebCore::FormData>&&);
GBytes* webkitFormDataInputStreamReadAll(WebKitFormDataInputStream*);
