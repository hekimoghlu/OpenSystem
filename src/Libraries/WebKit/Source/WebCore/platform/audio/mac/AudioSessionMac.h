/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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

#if USE(AUDIO_SESSION) && PLATFORM(MAC)

#include "AudioSessionCocoa.h"
#include <pal/spi/cf/CoreAudioSPI.h>
#include <wtf/BlockPtr.h>
#include <wtf/TZoneMalloc.h>

typedef UInt32 AudioObjectID;
typedef struct AudioObjectPropertyAddress AudioObjectPropertyAddress;

namespace WebCore {

class AudioSessionMac final : public AudioSessionCocoa {
    WTF_MAKE_TZONE_ALLOCATED(AudioSessionMac);
public:
    static Ref<AudioSessionMac> create();
    ~AudioSessionMac();

private:
    AudioSessionMac();

    void addSampleRateObserverIfNeeded() const;
    void addBufferSizeObserverIfNeeded() const;
    void addDefaultDeviceObserverIfNeeded() const;
    void addMuteChangeObserverIfNeeded() const;
    void removeMuteChangeObserverIfNeeded() const;

    float sampleRateWithoutCaching() const;
    std::optional<size_t> bufferSizeWithoutCaching() const;
    void removePropertyListenersForDefaultDevice() const;

    void handleDefaultDeviceChange();
    void handleSampleRateChange() const;
    void handleBufferSizeChange() const;

    AudioDeviceID defaultDevice() const;
    static const AudioObjectPropertyAddress& defaultOutputDeviceAddress();
    static const AudioObjectPropertyAddress& nominalSampleRateAddress();
    static const AudioObjectPropertyAddress& bufferSizeAddress();
    static const AudioObjectPropertyAddress& muteAddress();

    bool hasSampleRateObserver() const { return !!m_handleSampleRateChangeBlock; };
    bool hasBufferSizeObserver() const { return !!m_handleBufferSizeChangeBlock; };
    bool hasDefaultDeviceObserver() const { return !!m_handleDefaultDeviceChangeBlock; };
    bool hasMuteChangeObserver() const { return !!m_handleMutedStateChangeBlock; };

    // AudioSession
    CategoryType category() const final { return m_category; }
    RouteSharingPolicy routeSharingPolicy() const { return m_policy; }
    void audioOutputDeviceChanged() final;
    void setIsPlayingToBluetoothOverride(std::optional<bool>) final;
    void setCategory(CategoryType, Mode, RouteSharingPolicy) final;
    float sampleRate() const final;
    size_t bufferSize() const final;
    size_t numberOfOutputChannels() const final;
    size_t maximumNumberOfOutputChannels() const final;
    String routingContextUID() const final;
    size_t preferredBufferSize() const final;
    void setPreferredBufferSize(size_t) final;
    size_t outputLatency() const final;
    bool isMuted() const final;
    void handleMutedStateChange() final;
    void addConfigurationChangeObserver(AudioSessionConfigurationChangeObserver&) final;
    void removeConfigurationChangeObserver(AudioSessionConfigurationChangeObserver&) final;

    WTFLogChannel& logChannel() const;
    uint64_t logIdentifier() const;

    std::optional<bool> m_lastMutedState;
    mutable WeakHashSet<AudioSessionConfigurationChangeObserver> m_configurationChangeObservers;
    AudioSession::CategoryType m_category { AudioSession::CategoryType::None };
    RouteSharingPolicy m_policy { RouteSharingPolicy::Default };
#if ENABLE(ROUTING_ARBITRATION)
    bool m_setupArbitrationOngoing { false };
    bool m_inRoutingArbitration { false };
    std::optional<bool> m_playingToBluetooth;
    std::optional<bool> m_playingToBluetoothOverride;
#endif
    mutable std::optional<double> m_sampleRate;
    mutable std::optional<size_t> m_bufferSize;
    mutable std::optional<AudioDeviceID> m_defaultDevice;

    mutable BlockPtr<void(unsigned, const struct AudioObjectPropertyAddress*)> m_handleDefaultDeviceChangeBlock;
    mutable BlockPtr<void(unsigned, const struct AudioObjectPropertyAddress*)> m_handleSampleRateChangeBlock;
    mutable BlockPtr<void(unsigned, const struct AudioObjectPropertyAddress*)> m_handleBufferSizeChangeBlock;
    mutable BlockPtr<void(unsigned, const struct AudioObjectPropertyAddress*)> m_handleMutedStateChangeBlock;
};

}

#endif
