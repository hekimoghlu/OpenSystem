/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#include "config.h"
#include "GStreamerQuirkOpenMAX.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_openmax_quirks_debug);
#define GST_CAT_DEFAULT webkit_openmax_quirks_debug

GStreamerQuirkOpenMAX::GStreamerQuirkOpenMAX()
{
    GST_DEBUG_CATEGORY_INIT(webkit_openmax_quirks_debug, "webkitquirksopenmax", 0, "WebKit OpenMAX Quirks");
}

bool GStreamerQuirkOpenMAX::processWebAudioSilentBuffer(GstBuffer* buffer) const
{
    GST_TRACE("Force disabling GAP buffer flag");
    GST_BUFFER_FLAG_UNSET(buffer, GST_BUFFER_FLAG_GAP);
    GST_BUFFER_FLAG_UNSET(buffer, GST_BUFFER_FLAG_DROPPABLE);
    return true;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
