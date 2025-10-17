/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#if ENABLE(VIDEO) && USE(GSTREAMER) && ENABLE(MEDIA_SOURCE)

#include "GStreamerCommon.h"
#include "MediaDescription.h"

#include <gst/gst.h>
#include <wtf/text/StringView.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GStreamerMediaDescription : public MediaDescription {
public:
    static Ref<GStreamerMediaDescription> create(const GRefPtr<GstCaps>& caps)
    {
        return adoptRef(*new GStreamerMediaDescription(caps));
    }

    virtual ~GStreamerMediaDescription() = default;

    bool isVideo() const final;
    bool isAudio() const final;
    bool isText() const final;

private:
    GStreamerMediaDescription(const GRefPtr<GstCaps>& caps)
        : MediaDescription(extractCodecName(caps))
        , m_caps(caps)
    {
    }

    String extractCodecName(const GRefPtr<GstCaps>&) const;
    const GRefPtr<GstCaps> m_caps;
};

} // namespace WebCore.

#endif // USE(GSTREAMER)
