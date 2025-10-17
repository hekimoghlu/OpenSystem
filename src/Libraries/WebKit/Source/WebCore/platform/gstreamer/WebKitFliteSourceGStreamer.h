/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#if USE(FLITE) && USE(GSTREAMER)

#include <gst/gst.h>
#include <wtf/Forward.h>

namespace WebCore {
class PlatformSpeechSynthesisVoice;
}

#define WEBKIT_TYPE_FLITE_SRC (webkit_flite_src_get_type())
#define WEBKIT_FLITE_SRC(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_FLITE_SRC, WebKitFliteSrc))

typedef struct _WebKitFliteSrc WebKitFliteSrc;

GType webkit_flite_src_get_type();

Vector<Ref<WebCore::PlatformSpeechSynthesisVoice>>& ensureFliteVoicesInitialized();
void webKitFliteSrcSetUtterance(WebKitFliteSrc*, const WebCore::PlatformSpeechSynthesisVoice*, const String&);

#endif // USE(FLITE) && USE(GSTREAMER)
