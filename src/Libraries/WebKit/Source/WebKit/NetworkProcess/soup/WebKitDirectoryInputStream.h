/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

#define WEBKIT_TYPE_DIRECTORY_INPUT_STREAM            (webkit_directory_input_stream_get_type ())
#define WEBKIT_DIRECTORY_INPUT_STREAM(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_TYPE_DIRECTORY_INPUT_STREAM, WebKitDirectoryInputStream))
#define WEBKIT_IS_DIRECTORY_INPUT_STREAM(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_TYPE_DIRECTORY_INPUT_STREAM))
#define WEBKIT_DIRECTORY_INPUT_STREAM_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), WEBKIT_TYPE_DIRECTORY_INPUT_STREAM, WebKitDirectoryInputStreamClass))
#define WEBKIT_IS_DIRECTORY_INPUT_STREAM_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((obj), WEBKIT_TYPE_DIRECTORY_INPUT_STREAM))
#define WEBKIT_DIRECTORY_INPUT_STREAM_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), WEBKIT_TYPE_DIRECTORY_INPUT_STREAM, WebKitDirectoryInputStreamClass))

typedef struct _WebKitDirectoryInputStream WebKitDirectoryInputStream;
typedef struct _WebKitDirectoryInputStreamClass WebKitDirectoryInputStreamClass;
typedef struct _WebKitDirectoryInputStreamPrivate WebKitDirectoryInputStreamPrivate;

struct _WebKitDirectoryInputStream {
    GInputStream parent;

    WebKitDirectoryInputStreamPrivate* priv;
};

struct _WebKitDirectoryInputStreamClass {
    GInputStreamClass parentClass;
};

GType webkit_directory_input_stream_get_type(void);

GRefPtr<GInputStream> webkitDirectoryInputStreamNew(GRefPtr<GFileEnumerator>&&, CString&& uri);
