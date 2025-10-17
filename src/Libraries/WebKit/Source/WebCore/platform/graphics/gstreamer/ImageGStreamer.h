/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

#include "FloatRect.h"
#include "GStreamerCommon.h"
#include "PlatformImage.h"

#include <gst/video/video-frame.h>

#include <wtf/Forward.h>

namespace WebCore {
class IntSize;

class ImageGStreamer : public RefCounted<ImageGStreamer> {
public:
    static Ref<ImageGStreamer> create(GRefPtr<GstSample>&& sample)
    {
        return adoptRef(*new ImageGStreamer(WTFMove(sample)));
    }
    ~ImageGStreamer();

    operator bool() const { return !!m_image; }

    PlatformImagePtr image() const { return m_image; }

    void setCropRect(FloatRect rect) { m_cropRect = rect; }
    FloatRect rect()
    {
        ASSERT(m_image);
        if (!m_cropRect.isEmpty())
            return FloatRect(m_cropRect);
        return FloatRect(0, 0, m_size.width(), m_size.height());
    }

    bool hasAlpha() const { return m_hasAlpha; }

private:
    ImageGStreamer(GRefPtr<GstSample>&&);
    GRefPtr<GstSample> m_sample;
    PlatformImagePtr m_image;
    FloatRect m_cropRect;
#if USE(CAIRO)
    GstVideoFrame m_videoFrame;
    bool m_frameMapped { false };
#endif
    FloatSize m_size;
    bool m_hasAlpha { false };
};

}
#endif // ENABLE(VIDEO) && USE(GSTREAMER)
