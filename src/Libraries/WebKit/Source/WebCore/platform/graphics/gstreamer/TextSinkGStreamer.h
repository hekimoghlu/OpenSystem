/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
#include <wtf/Forward.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {
class MediaPlayerPrivateGStreamer;
}

G_BEGIN_DECLS

#define WEBKIT_TYPE_TEXT_SINK webkit_text_sink_get_type()
GType webkit_text_sink_get_type(void);

#define WEBKIT_TEXT_SINK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_TEXT_SINK, WebKitTextSink))
#define WEBKIT_TEXT_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_TEXT_SINK, WebKitTextSinkClass))
#define WEBKIT_IS_TEXT_SINK(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_TEXT_SINK))
#define WEBKIT_IS_TEXT_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_TEXT_SINK))
#define WEBKIT_TEXT_SINK_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), WEBKIT_TYPE_TEXT_SINK, WebKitTextSinkClass))

typedef struct _WebKitTextSink WebKitTextSink;
typedef struct _WebKitTextSinkClass WebKitTextSinkClass;
typedef struct _WebKitTextSinkPrivate WebKitTextSinkPrivate;

struct _WebKitTextSink {
    GstBin parent;
    WebKitTextSinkPrivate* priv;
};

struct _WebKitTextSinkClass {
    GstBinClass parentClass;
};

GstElement* webkitTextSinkNew(ThreadSafeWeakPtr<WebCore::MediaPlayerPrivateGStreamer>&&);

G_END_DECLS

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
