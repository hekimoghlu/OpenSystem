/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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

#include <gst/app/gstappsink.h>
#include <gst/base/gstbasesink.h>

namespace WebCore {

void registerAppsinkWithWorkaroundsIfNeeded();

// Workaround for: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/merge_requests/3471 which will land in GStreamer 1.24.0.
// basesink: Support position queries after non-resetting flushes
void installBaseSinkPositionFlushWorkaroundIfNeeded(GstBaseSink* basesink);

} // namespace WebCore

#define WEBKIT_TYPE_APP_SINK_WITH_WORKAROUNDS (webkit_app_sink_with_workarounds_get_type())

#define WEBKIT_APP_SINK_WITH_WORKAROUNDS(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_APP_SINK_WITH_WORKAROUNDS, WebKitAppSinkWithWorkaround))
#define WEBKIT_APP_SINK_WITH_WORKAROUNDS_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_APP_SINK_WITH_WORKAROUNDS, WebKitAppSinkWithWorkaroundClass))
#define WEBKIT_IS_APP_SINK_WITH_WORKAROUNDS(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_APP_SINK_WITH_WORKAROUNDS))
#define WEBKIT_IS_APP_SINK_WITH_WORKAROUNDS_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_APP_SINK_WITH_WORKAROUNDS))
#define WEBKIT_APP_SINK_WITH_WORKAROUNDS_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), WEBKIT_TYPE_APP_SINK_WITH_WORKAROUNDS, WebKitAppSinkWithWorkaroundClass))

struct WebKitAppSinkWithWorkaroundsPrivate;
struct WebKitAppSinkWithWorkarounds {
    GstAppSink parent;
    WebKitAppSinkWithWorkaroundsPrivate* priv;
};

struct WebKitAppSinkWithWorkaroundsClass {
    GstAppSinkClass parent;
};

GType webkit_app_sink_with_workarounds_get_type();

#endif // USE(GSTREAMER)
