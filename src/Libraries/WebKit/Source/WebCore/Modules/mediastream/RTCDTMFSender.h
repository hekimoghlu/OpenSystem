/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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

#if ENABLE(WEB_RTC)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include "Timer.h"

namespace WebCore {

class MediaStreamTrack;
class RTCDTMFSenderBackend;
class RTCRtpSender;

class RTCDTMFSender final : public RefCounted<RTCDTMFSender>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCDTMFSender);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<RTCDTMFSender> create(ScriptExecutionContext&, RTCRtpSender&, std::unique_ptr<RTCDTMFSenderBackend>&&);
    virtual ~RTCDTMFSender();

    bool canInsertDTMF() const;
    String toneBuffer() const;

    ExceptionOr<void> insertDTMF(const String& tones, size_t duration, size_t interToneGap);

private:
    RTCDTMFSender(ScriptExecutionContext&, RTCRtpSender&, std::unique_ptr<RTCDTMFSenderBackend>&&);

    // ActiveDOMObject.
    void stop() final;

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RTCDTMFSender; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    bool virtualHasPendingActivity() const final { return m_isPendingPlayoutTask; }

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    bool isStopped() const { return !m_sender; }

    void playNextTone();
    void onTonePlayed();
    void toneTimerFired();

    Timer m_toneTimer;
    WeakPtr<RTCRtpSender> m_sender;
    std::unique_ptr<RTCDTMFSenderBackend> m_backend;
    String m_tones;
    size_t m_duration;
    size_t m_interToneGap;
    bool m_isPendingPlayoutTask { false };
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
