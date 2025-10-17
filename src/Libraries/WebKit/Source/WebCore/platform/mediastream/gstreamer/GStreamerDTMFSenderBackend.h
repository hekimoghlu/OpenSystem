/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "RTCDTMFSenderBackend.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

// Use eager initialization for the WeakPtrFactory since we call makeWeakPtr() from another thread.
class GStreamerDTMFSenderBackend final : public RTCDTMFSenderBackend, public CanMakeWeakPtr<GStreamerDTMFSenderBackend, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerDTMFSenderBackend);
public:
    explicit GStreamerDTMFSenderBackend();
    ~GStreamerDTMFSenderBackend();

private:
    // RTCDTMFSenderBackend
    bool canInsertDTMF() final;
    void playTone(const char tone, size_t duration, size_t interToneGap) final;
    void onTonePlayed(Function<void()>&&) final;
    String tones() const final;
    size_t duration() const final;
    size_t interToneGap() const final;

    Function<void()> m_onTonePlayed;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
