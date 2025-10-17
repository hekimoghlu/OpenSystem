/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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

#include "PlatformMediaResourceLoader.h"
#include <gst/base/gstpushsrc.h>
#include <gst/gst.h>
#include <wtf/Forward.h>

namespace WebCore {
class MediaPlayer;
class SecurityOrigin;
}

G_BEGIN_DECLS

#define WEBKIT_TYPE_WEB_SRC            (webkit_web_src_get_type ())
#define WEBKIT_WEB_SRC(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), WEBKIT_TYPE_WEB_SRC, WebKitWebSrc))
#define WEBKIT_WEB_SRC_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), WEBKIT_TYPE_WEB_SRC, WebKitWebSrcClass))
#define WEBKIT_IS_WEB_SRC(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), WEBKIT_TYPE_WEB_SRC))
#define WEBKIT_IS_WEB_SRC_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), WEBKIT_TYPE_WEB_SRC))
#define WEBKIT_WEB_SRC_CAST(obj)       ((WebKitWebSrc*)(obj))

#define WEBKIT_WEB_SRC_RESOURCE_LOADER_CONTEXT_TYPE_NAME  "webkit.resource-loader"

struct WebKitWebSrcPrivate;

struct WebKitWebSrc {
    GstPushSrc parent;

    WebKitWebSrcPrivate *priv;
};

struct WebKitWebSrcClass {
    GstPushSrcClass parentClass;
};

GType webkit_web_src_get_type(void);
void webKitWebSrcSetResourceLoader(WebKitWebSrc*, WebCore::PlatformMediaResourceLoader&);
void webKitWebSrcSetReferrer(WebKitWebSrc*, const String&);
bool webKitSrcPassedCORSAccessCheck(WebKitWebSrc*);
bool webKitSrcIsCrossOrigin(WebKitWebSrc*, const WebCore::SecurityOrigin&);

G_END_DECLS

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
