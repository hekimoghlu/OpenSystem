/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

#if ENABLE(VIDEO) && ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "MediaPlayer.h"
#include "MediaSource.h"
#include "MediaSourcePrivate.h"
#include "SourceBufferPrivate.h"
#include "SourceBufferPrivateClient.h"
#include "SourceBufferPrivateGStreamer.h"

namespace WebCore {

class MediaPlayerPrivateGStreamerMSE;

} // namespace WebCore

G_BEGIN_DECLS

#define WEBKIT_TYPE_MEDIA_SRC            (webkit_media_src_get_type ())
#define WEBKIT_MEDIA_SRC(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_TYPE_MEDIA_SRC, WebKitMediaSrc))
#define WEBKIT_MEDIA_SRC_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), WEBKIT_TYPE_MEDIA_SRC, WebKitMediaSrcClass))
#define WEBKIT_IS_MEDIA_SRC(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_TYPE_MEDIA_SRC))
#define WEBKIT_IS_MEDIA_SRC_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), WEBKIT_TYPE_MEDIA_SRC))

struct WebKitMediaSrcPrivate;

struct WebKitMediaSrc {
    GstElement parent;

    WebKitMediaSrcPrivate *priv;
};

struct WebKitMediaSrcClass {
    GstElementClass parentClass;
};

GType webkit_media_src_get_type(void);

void webKitMediaSrcEmitStreams(WebKitMediaSrc* source, const Vector<RefPtr<WebCore::MediaSourceTrackGStreamer>>& tracks);

void webKitMediaSrcFlush(WebKitMediaSrc*, TrackID streamId);

void webKitMediaSrcSetPlayer(WebKitMediaSrc*, ThreadSafeWeakPtr<WebCore::MediaPlayerPrivateGStreamerMSE>&&);

G_END_DECLS

namespace WTF {
template<> GRefPtr<WebKitMediaSrc> adoptGRef(WebKitMediaSrc* ptr);
template<> WebKitMediaSrc* refGPtr<WebKitMediaSrc>(WebKitMediaSrc* ptr);
template<> void derefGPtr<WebKitMediaSrc>(WebKitMediaSrc* ptr);
} // namespace WTF

#endif // USE(GSTREAMER)
