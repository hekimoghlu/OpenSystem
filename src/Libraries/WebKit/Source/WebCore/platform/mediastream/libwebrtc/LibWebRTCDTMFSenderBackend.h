/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "RTCDTMFSenderBackend.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/dtmf_sender_interface.h>
#include <webrtc/api/scoped_refptr.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {
class LibWebRTCDTMFSenderBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::LibWebRTCDTMFSenderBackend> : std::true_type { };
}

namespace WebCore {

// Use eager initialization for the WeakPtrFactory since we construct WeakPtrs on another thread.
class LibWebRTCDTMFSenderBackend final : public RTCDTMFSenderBackend, private webrtc::DtmfSenderObserverInterface, public CanMakeWeakPtr<LibWebRTCDTMFSenderBackend, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCDTMFSenderBackend);
public:
    explicit LibWebRTCDTMFSenderBackend(rtc::scoped_refptr<webrtc::DtmfSenderInterface>&&);
    ~LibWebRTCDTMFSenderBackend();

private:
    // RTCDTMFSenderBackend
    bool canInsertDTMF() final;
    void playTone(const char tone, size_t duration, size_t interToneGap) final;
    void onTonePlayed(Function<void()>&&) final;
    String tones() const final;
    size_t duration() const final;
    size_t interToneGap() const final;

    // DtmfSenderObserverInterface
    void OnToneChange(const std::string& tone, const std::string&) final;

    rtc::scoped_refptr<webrtc::DtmfSenderInterface> m_sender;
    Function<void()> m_onTonePlayed;
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
