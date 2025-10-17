/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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

namespace WebCore {
class MediaPlayerPrivateGStreamer;
}

G_BEGIN_DECLS

#define WEBKIT_TYPE_GL_VIDEO_SINK            (webkit_gl_video_sink_get_type ())
#define WEBKIT_GL_VIDEO_SINK(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_TYPE_GL_VIDEO_SINK, WebKitGLVideoSink))
#define WEBKIT_GL_VIDEO_SINK_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), WEBKIT_TYPE_GL_VIDEO_SINK, WebKitGLVideoSinkClass))
#define WEBKIT_IS_GL_VIDEO_SINK(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_TYPE_GL_VIDEO_SINK))
#define WEBKIT_IS_GL_VIDEO_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), WEBKIT_TYPE_GL_VIDEO_SINK))

typedef struct _WebKitGLVideoSink        WebKitGLVideoSink;
typedef struct _WebKitGLVideoSinkClass   WebKitGLVideoSinkClass;
typedef struct _WebKitGLVideoSinkPrivate WebKitGLVideoSinkPrivate;

struct _WebKitGLVideoSink {
    GstBin parent;

    WebKitGLVideoSinkPrivate *priv;
};

struct _WebKitGLVideoSinkClass {
    GstBinClass parentClass;
};

GType webkit_gl_video_sink_get_type(void);

bool webKitGLVideoSinkProbePlatform();
void webKitGLVideoSinkSetMediaPlayerPrivate(WebKitGLVideoSink*, WebCore::MediaPlayerPrivateGStreamer*);

G_END_DECLS

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
