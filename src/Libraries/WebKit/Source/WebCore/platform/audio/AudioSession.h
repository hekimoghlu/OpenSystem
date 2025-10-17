/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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

#if USE(AUDIO_SESSION)

#include <memory>
#include <wtf/CompletionHandler.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/Observer.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class AudioSessionInterruptionObserver;
class AudioSessionRoutingArbitrationClient;
class AudioSessionConfigurationChangeObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AudioSessionInterruptionObserver> : std::true_type { };
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AudioSessionRoutingArbitrationClient> : std::true_type { };
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AudioSessionConfigurationChangeObserver> : std::true_type { };
}

namespace WTF {
class Logger;
}

namespace WebCore {

enum class RouteSharingPolicy : uint8_t {
    Default,
    LongFormAudio,
    Independent,
    LongFormVideo
};

enum class AudioSessionCategory : uint8_t {
    None,
    AmbientSound,
    SoloAmbientSound,
    MediaPlayback,
    RecordAudio,
    PlayAndRecord,
    AudioProcessing,
};

enum class AudioSessionMode : uint8_t {
    // FIXME: This is not exhaustive.
    Default,
    VideoChat,
    MoviePlayback,
};

enum class AudioSessionSoundStageSize : uint8_t {
    Automatic,
    Small,
    Medium,
    Large,
};

class AudioSession;
class AudioSessionRoutingArbitrationClient;
class AudioSessionInterruptionObserver;

class AudioSessionConfigurationChangeObserver : public CanMakeWeakPtr<AudioSessionConfigurationChangeObserver> {
public:
    virtual ~AudioSessionConfigurationChangeObserver() = default;

    virtual void hardwareMutedStateDidChange(const AudioSession&) = 0;
    virtual void bufferSizeDidChange(const AudioSession&) { }
    virtual void sampleRateDidChange(const AudioSession&) { }
};

class WEBCORE_EXPORT AudioSession : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<AudioSession> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AudioSession, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(AudioSession);
public:
    static Ref<AudioSession> create();
    static void setSharedSession(Ref<AudioSession>&&);
    static AudioSession& sharedSession();
    static Ref<AudioSession> protectedSharedSession() { return sharedSession(); }

    static bool enableMediaPlayback();

    using ChangedObserver = WTF::Observer<void(AudioSession&)>;
    static void addAudioSessionChangedObserver(const ChangedObserver&);

    virtual ~AudioSession();

    using CategoryType = AudioSessionCategory;
    virtual CategoryType category() const;
    using Mode = AudioSessionMode;
    virtual Mode mode() const;
    virtual void setCategory(CategoryType, Mode, RouteSharingPolicy);

    virtual void setCategoryOverride(CategoryType);
    virtual CategoryType categoryOverride() const;

    virtual RouteSharingPolicy routeSharingPolicy() const;
    virtual String routingContextUID() const;

    virtual float sampleRate() const;
    virtual size_t bufferSize() const;
    virtual size_t numberOfOutputChannels() const;
    virtual size_t maximumNumberOfOutputChannels() const;

    bool tryToSetActive(bool);

    virtual size_t preferredBufferSize() const;
    virtual void setPreferredBufferSize(size_t);

    virtual size_t outputLatency() const { return 0; }

    virtual void addConfigurationChangeObserver(AudioSessionConfigurationChangeObserver&);
    virtual void removeConfigurationChangeObserver(AudioSessionConfigurationChangeObserver&);

    virtual void audioOutputDeviceChanged();
    virtual void setIsPlayingToBluetoothOverride(std::optional<bool>);

    virtual bool isMuted() const { return false; }
    virtual void handleMutedStateChange() { }

    virtual void beginInterruption();
    enum class MayResume : bool { No, Yes };
    virtual void endInterruption(MayResume);

    virtual void beginInterruptionForTesting() { beginInterruption(); }
    virtual void endInterruptionForTesting() { endInterruption(MayResume::Yes); }
    virtual void clearInterruptionFlagForTesting() { }

    virtual void addInterruptionObserver(AudioSessionInterruptionObserver&);
    virtual void removeInterruptionObserver(AudioSessionInterruptionObserver&);

    virtual bool isActive() const { return m_active; }

    virtual void setRoutingArbitrationClient(WeakPtr<AudioSessionRoutingArbitrationClient>&& client) { m_routingArbitrationClient = client; }

    static bool shouldManageAudioSessionCategory();
    static void setShouldManageAudioSessionCategory(bool);

    virtual void setHostProcessAttribution(audit_token_t) { };
    virtual void setPresentingProcesses(Vector<audit_token_t>&&) { };

    bool isInterrupted() const { return m_isInterrupted; }

    virtual void setSceneIdentifier(const String&) { }
    virtual const String& sceneIdentifier() const { return nullString(); }

    using SoundStageSize = AudioSessionSoundStageSize;
    virtual void setSoundStageSize(SoundStageSize) { }
    virtual SoundStageSize soundStageSize() const { return SoundStageSize::Automatic; }

protected:
    friend class NeverDestroyed<AudioSession>;
    AudioSession();

    virtual bool tryToSetActiveInternal(bool);
    void activeStateChanged();

    Logger& logger();
    ASCIILiteral logClassName() const { return "AudioSession"_s; }
    WTFLogChannel& logChannel() const;
    uint64_t logIdentifier() const { return 0; }

    mutable RefPtr<Logger> m_logger;

    WeakHashSet<AudioSessionInterruptionObserver> m_interruptionObservers;

    WeakPtr<AudioSessionRoutingArbitrationClient> m_routingArbitrationClient;
    AudioSession::CategoryType m_categoryOverride { AudioSession::CategoryType::None };
    bool m_active { false }; // Used only for testing.
    bool m_isInterrupted { false };
};

class AudioSessionInterruptionObserver : public CanMakeWeakPtr<AudioSessionInterruptionObserver> {
public:
    virtual ~AudioSessionInterruptionObserver() = default;

    virtual void beginAudioSessionInterruption() = 0;
    virtual void endAudioSessionInterruption(AudioSession::MayResume) = 0;
    virtual void audioSessionActiveStateChanged() { }
};

enum class AudioSessionRoutingArbitrationError : uint8_t { None, Failed, Cancelled };

class WEBCORE_EXPORT AudioSessionRoutingArbitrationClient : public CanMakeWeakPtr<AudioSessionRoutingArbitrationClient> {
public:
    USING_CAN_MAKE_WEAKPTR(CanMakeWeakPtr<AudioSessionRoutingArbitrationClient>);

    virtual ~AudioSessionRoutingArbitrationClient() = default;
    using RoutingArbitrationError = AudioSessionRoutingArbitrationError;

    enum class DefaultRouteChanged : bool { No, Yes };

    using ArbitrationCallback = CompletionHandler<void(RoutingArbitrationError, DefaultRouteChanged)>;

    virtual void beginRoutingArbitrationWithCategory(AudioSession::CategoryType, ArbitrationCallback&&) = 0;
    virtual void leaveRoutingAbritration() = 0;

    virtual uint64_t logIdentifier() const = 0;
    virtual bool canLog() const = 0;
};

WEBCORE_EXPORT String convertEnumerationToString(RouteSharingPolicy);
WEBCORE_EXPORT String convertEnumerationToString(AudioSession::CategoryType);
WEBCORE_EXPORT String convertEnumerationToString(AudioSession::Mode);
WEBCORE_EXPORT String convertEnumerationToString(AudioSessionRoutingArbitrationClient::RoutingArbitrationError);
WEBCORE_EXPORT String convertEnumerationToString(AudioSessionRoutingArbitrationClient::DefaultRouteChanged);
WEBCORE_EXPORT String convertEnumerationToString(AudioSession::SoundStageSize);

} // namespace WebCore

namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebCore::RouteSharingPolicy> {
    static String toString(const WebCore::RouteSharingPolicy policy)
    {
        return convertEnumerationToString(policy);
    }
};

template <>
struct LogArgument<WebCore::AudioSession::CategoryType> {
    static String toString(const WebCore::AudioSession::CategoryType category)
    {
        return convertEnumerationToString(category);
    }
};

template <>
struct LogArgument<WebCore::AudioSession::Mode> {
    static String toString(const WebCore::AudioSession::Mode mode)
    {
        return convertEnumerationToString(mode);
    }
};

template <>
struct LogArgument<WebCore::AudioSessionRoutingArbitrationClient::RoutingArbitrationError> {
    static String toString(const WebCore::AudioSessionRoutingArbitrationClient::RoutingArbitrationError error)
    {
        return convertEnumerationToString(error);
    }
};

template <>
struct LogArgument<WebCore::AudioSessionRoutingArbitrationClient::DefaultRouteChanged> {
    static String toString(const WebCore::AudioSessionRoutingArbitrationClient::DefaultRouteChanged changed)
    {
        return convertEnumerationToString(changed);
    }
};

template <>
struct LogArgument<WebCore::AudioSession::SoundStageSize> {
    static String toString(const WebCore::AudioSession::SoundStageSize size)
    {
        return convertEnumerationToString(size);
    }
};

} // namespace WTF

#endif // USE(AUDIO_SESSION)
