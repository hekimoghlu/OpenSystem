/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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

#include <glib-object.h>
#include <gst/video/gstvideosink.h>

#define WEBKIT_TYPE_VIDEO_SINK webkit_video_sink_get_type()

#define WEBKIT_VIDEO_SINK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_VIDEO_SINK, WebKitVideoSink))
#define WEBKIT_VIDEO_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_VIDEO_SINK, WebKitVideoSinkClass))
#define WEBKIT_IS_VIDEO_SINK(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_VIDEO_SINK))
#define WEBKIT_IS_VIDEO_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_VIDEO_SINK))
#define WEBKIT_VIDEO_SINK_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), WEBKIT_TYPE_VIDEO_SINK, WebKitVideoSinkClass))

typedef struct _WebKitVideoSink WebKitVideoSink;
typedef struct _WebKitVideoSinkClass WebKitVideoSinkClass;
typedef struct _WebKitVideoSinkPrivate WebKitVideoSinkPrivate;

struct _WebKitVideoSink {
    GstVideoSink parent;
    WebKitVideoSinkPrivate* priv;
};

struct _WebKitVideoSinkClass {
    GstVideoSinkClass parentClass;
};

GType webkit_video_sink_get_type() G_GNUC_CONST;

GstElement* webkitVideoSinkNew();

#endif // USE(GSTREAMER)
