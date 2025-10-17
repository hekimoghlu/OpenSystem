/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#include "GStreamerDTMFSenderBackend.h"

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "NotImplemented.h"
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GStreamerDTMFSenderBackend);

GStreamerDTMFSenderBackend::GStreamerDTMFSenderBackend()
{
    notImplemented();
}

GStreamerDTMFSenderBackend::~GStreamerDTMFSenderBackend()
{
    notImplemented();
}

bool GStreamerDTMFSenderBackend::canInsertDTMF()
{
    notImplemented();
    return false;
}

void GStreamerDTMFSenderBackend::playTone(const char, size_t, size_t)
{
    notImplemented();
}

String GStreamerDTMFSenderBackend::tones() const
{
    notImplemented();
    return { };
}

size_t GStreamerDTMFSenderBackend::duration() const
{
    notImplemented();
    return 0;
}

size_t GStreamerDTMFSenderBackend::interToneGap() const
{
    notImplemented();
    return 0;
}

void GStreamerDTMFSenderBackend::onTonePlayed(Function<void()>&& onTonePlayed)
{
    m_onTonePlayed = WTFMove(onTonePlayed);
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
