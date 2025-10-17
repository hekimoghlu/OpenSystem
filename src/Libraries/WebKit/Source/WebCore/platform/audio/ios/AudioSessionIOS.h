/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#if USE(AUDIO_SESSION) && PLATFORM(IOS_FAMILY)

#include "AudioSessionCocoa.h"
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebInterruptionObserverHelper;

namespace WTF {
class WorkQueue;
}

namespace WebCore {

class AudioSessionIOS final : public AudioSessionCocoa {
    WTF_MAKE_TZONE_ALLOCATED(AudioSessionIOS);
public:
    static Ref<AudioSessionIOS> create();
    virtual ~AudioSessionIOS();

    void setHostProcessAttribution(audit_token_t) final;
    void setPresentingProcesses(Vector<audit_token_t>&&) final;

    using CategoryChangedObserver = WTF::Observer<void(AudioSession&, CategoryType)>;
    WEBCORE_EXPORT static void addAudioSessionCategoryChangedObserver(const CategoryChangedObserver&);

private:
    AudioSessionIOS();

    // AudioSession
    CategoryType category() const final;
    Mode mode() const final;
    void setCategory(CategoryType, Mode, RouteSharingPolicy) final;
    float sampleRate() const final;
    size_t bufferSize() const final;
    size_t numberOfOutputChannels() const final;
    size_t maximumNumberOfOutputChannels() const final;
    RouteSharingPolicy routeSharingPolicy() const final;
    String routingContextUID() const final;
    size_t preferredBufferSize() const final;
    void setPreferredBufferSize(size_t) final;
    size_t outputLatency() const final;
    bool isMuted() const final;
    void handleMutedStateChange() final;

    void updateSpatialExperience();

    void setSceneIdentifier(const String&) final;
    const String& sceneIdentifier() const final { return m_sceneIdentifier; }

    void setSoundStageSize(SoundStageSize) final;
    SoundStageSize soundStageSize() const final { return m_soundStageSize; }

    String m_lastSetPreferredMicrophoneID;
    RetainPtr<WebInterruptionObserverHelper> m_interruptionObserverHelper;
    String m_sceneIdentifier;
    SoundStageSize m_soundStageSize { SoundStageSize::Automatic };
};

}

#endif
