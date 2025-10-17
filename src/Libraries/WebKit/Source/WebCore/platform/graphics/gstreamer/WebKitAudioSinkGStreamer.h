/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#if USE(GSTREAMER)

#include <gst/gst.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_AUDIO_SINK            (webkit_audio_sink_get_type())
#define WEBKIT_AUDIO_SINK(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_AUDIO_SINK, WebKitAudioSink))
#define WEBKIT_AUDIO_SINK_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_AUDIO_SINK, WebKitAudioSinkClass))
#define WEBKIT_IS_AUDIO_SINK(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_AUDIO_SINK))
#define WEBKIT_IS_AUDIO_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_AUDIO_SINK))

typedef struct _WebKitAudioSink        WebKitAudioSink;
typedef struct _WebKitAudioSinkClass   WebKitAudioSinkClass;
typedef struct _WebKitAudioSinkPrivate WebKitAudioSinkPrivate;

struct _WebKitAudioSink {
    GstBin parent;

    WebKitAudioSinkPrivate *priv;
};

struct _WebKitAudioSinkClass {
    GstBinClass parentClass;
};

GType webkit_audio_sink_get_type(void);

G_END_DECLS

GstElement* webkitAudioSinkNew();

#endif // USE(GSTREAMER)
