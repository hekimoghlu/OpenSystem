/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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

#if ENABLE(DOM_AUDIO_SESSION)

#include "ActiveDOMObject.h"
#include "AudioSession.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

enum class DOMAudioSessionType : uint8_t { Auto, Playback, Transient, TransientSolo, Ambient, PlayAndRecord };
enum class DOMAudioSessionState : uint8_t { Inactive, Active, Interrupted };

class DOMAudioSession final : public RefCounted<DOMAudioSession>, public ActiveDOMObject, public EventTarget, public AudioSessionInterruptionObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMAudioSession);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<DOMAudioSession> create(ScriptExecutionContext*);
    ~DOMAudioSession();

    using Type = DOMAudioSessionType;
    using State = DOMAudioSessionState;

    ExceptionOr<void> setType(Type);
    Type type() const;
    State state() const;

private:
    explicit DOMAudioSession(ScriptExecutionContext*);

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::DOMAudioSession; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // InterruptionObserver
    void beginAudioSessionInterruption() final;
    void endAudioSessionInterruption(AudioSession::MayResume) final;
    void audioSessionActiveStateChanged() final;

    void scheduleStateChangeEvent();

    bool m_hasScheduleStateChangeEvent { false };
    mutable std::optional<State> m_state;
};

} // namespace WebCore

#endif // ENABLE(DOM_AUDIO_SESSION)
