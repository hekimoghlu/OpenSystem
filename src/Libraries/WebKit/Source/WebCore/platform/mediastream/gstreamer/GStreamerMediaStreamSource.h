/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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

#if ENABLE(VIDEO) && ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

#include <gst/gst.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class MediaStreamPrivate;
class MediaStreamTrackPrivate;
}

#define WEBKIT_MEDIA_STREAM_SRC(o) (G_TYPE_CHECK_INSTANCE_CAST((o), WEBKIT_TYPE_MEDIA_STREAM_SRC, WebKitMediaStreamSrc))
#define WEBKIT_IS_MEDIA_STREAM_SRC(o) (G_TYPE_CHECK_INSTANCE_TYPE((o), WEBKIT_TYPE_MEDIA_STREAM_SRC))
#define WEBKIT_MEDIA_STREAM_SRC_CAST(o) ((WebKitMediaStreamSrc*) o)

#define WEBKIT_TYPE_MEDIA_STREAM_SRC (webkit_media_stream_src_get_type())
GType webkit_media_stream_src_get_type(void);

typedef struct _WebKitMediaStreamSrc WebKitMediaStreamSrc;
typedef struct _WebKitMediaStreamSrcClass WebKitMediaStreamSrcClass;
typedef struct _WebKitMediaStreamSrcPrivate WebKitMediaStreamSrcPrivate;

struct _WebKitMediaStreamSrc {
    GstBin parent;
    WebKitMediaStreamSrcPrivate* priv;
};

struct _WebKitMediaStreamSrcClass {
    GstBinClass parentClass;
};

GstElement* webkitMediaStreamSrcNew();
void webkitMediaStreamSrcSetStream(WebKitMediaStreamSrc*, WebCore::MediaStreamPrivate*, bool isVideoPlayer);
void webkitMediaStreamSrcAddTrack(WebKitMediaStreamSrc*, WebCore::MediaStreamTrackPrivate*, bool consumerIsVideoPlayer = false);
void webkitMediaStreamSrcReplaceTrack(WebKitMediaStreamSrc*, WTF::RefPtr<WebCore::MediaStreamTrackPrivate>&&);
void webkitMediaStreamSrcConfigureAudioTracks(WebKitMediaStreamSrc*, float volume, bool isMuted, bool isPlaying);
bool webkitMediaStreamSrcSignalEndOfStream(WebKitMediaStreamSrc*);

#endif // ENABLE(VIDEO) && ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
