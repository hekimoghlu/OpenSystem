/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#include "LibWebRTCDTMFSenderBackend.h"

#if USE(LIBWEBRTC)

#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCDTMFSenderBackend);

static inline String toWTFString(const std::string& value)
{
    return String::fromUTF8(value);
}

LibWebRTCDTMFSenderBackend::LibWebRTCDTMFSenderBackend(rtc::scoped_refptr<webrtc::DtmfSenderInterface>&& sender)
    : m_sender(WTFMove(sender))
{
    m_sender->RegisterObserver(this);
}

LibWebRTCDTMFSenderBackend::~LibWebRTCDTMFSenderBackend()
{
    m_sender->UnregisterObserver();
}

bool LibWebRTCDTMFSenderBackend::canInsertDTMF()
{
    return m_sender->CanInsertDtmf();
}

void LibWebRTCDTMFSenderBackend::playTone(const char tone, size_t duration, size_t interToneGap)
{
    bool ok = m_sender->InsertDtmf(std::string(1, tone), duration, interToneGap);
    ASSERT_UNUSED(ok, ok);
}

String LibWebRTCDTMFSenderBackend::tones() const
{
    return toWTFString(m_sender->tones());
}

size_t LibWebRTCDTMFSenderBackend::duration() const
{
    return m_sender->duration();
}

size_t LibWebRTCDTMFSenderBackend::interToneGap() const
{
    return m_sender->inter_tone_gap();
}

void LibWebRTCDTMFSenderBackend::OnToneChange(const std::string& tone, const std::string&)
{
    // We are just interested in notifying the end of the tone, which corresponds to the empty string.
    if (!tone.empty())
        return;
    callOnMainThread([this, weakThis = WeakPtr { *this }] {
        if (weakThis && m_onTonePlayed)
            m_onTonePlayed();
    });
}

void LibWebRTCDTMFSenderBackend::onTonePlayed(Function<void()>&& onTonePlayed)
{
    m_onTonePlayed = WTFMove(onTonePlayed);
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
