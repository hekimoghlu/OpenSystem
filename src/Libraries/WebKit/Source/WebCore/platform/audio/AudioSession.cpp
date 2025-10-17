/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "AudioSession.h"

#if USE(AUDIO_SESSION)

#include "Logging.h"
#include "NotImplemented.h"
#include <wtf/LoggerHelper.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(MAC)
#include "AudioSessionMac.h"
#endif

#if PLATFORM(IOS_FAMILY)
#include "AudioSessionIOS.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioSession);

std::atomic<bool> s_shouldManageAudioSessionCategory { false };
static bool s_mediaPlaybackEnabled { false };

bool AudioSession::shouldManageAudioSessionCategory()
{
    return s_shouldManageAudioSessionCategory.load();
}

void AudioSession::setShouldManageAudioSessionCategory(bool flag)
{
    s_shouldManageAudioSessionCategory.store(flag);
}

static RefPtr<AudioSession>& sharedAudioSession()
{
    static NeverDestroyed<RefPtr<AudioSession>> session;
    return session.get();
}

static Ref<AudioSession>& dummyAudioSession()
{
    static NeverDestroyed<Ref<AudioSession>> dummySession = AudioSession::create();
    return dummySession.get();
}

static WeakHashSet<AudioSession::ChangedObserver>& audioSessionChangedObservers()
{
    static NeverDestroyed<WeakHashSet<AudioSession::ChangedObserver>> observers;
    return observers;
}

bool AudioSession::enableMediaPlayback()
{
    if (s_mediaPlaybackEnabled)
        return false;

    s_mediaPlaybackEnabled = true;
    return true;
}

Ref<AudioSession> AudioSession::create()
{
#if PLATFORM(MAC)
    return AudioSessionMac::create();
#elif PLATFORM(IOS_FAMILY)
    return AudioSessionIOS::create();
#else
    return AudioSession::create();
#endif
}

AudioSession::AudioSession() = default;
AudioSession::~AudioSession() = default;

AudioSession& AudioSession::sharedSession()
{
    if (!s_mediaPlaybackEnabled)
        return dummyAudioSession();

    if (!sharedAudioSession())
        setSharedSession(AudioSession::create());

    return *sharedAudioSession();
}

void AudioSession::setSharedSession(Ref<AudioSession>&& session)
{
    sharedAudioSession() = session.copyRef();

    audioSessionChangedObservers().forEach([session] (auto& observer) {
        observer(session);
    });
}

void AudioSession::addAudioSessionChangedObserver(const ChangedObserver& observer)
{
    ASSERT(!audioSessionChangedObservers().contains(observer));
    audioSessionChangedObservers().add(observer);

    if (sharedAudioSession())
        observer(Ref { *sharedAudioSession() });
}

bool AudioSession::tryToSetActive(bool active)
{
    bool previousIsActive = isActive();
    if (!tryToSetActiveInternal(active))
        return false;

    ALWAYS_LOG(LOGIDENTIFIER, "is active = ", m_active, ", previousIsActive = ", previousIsActive);

    bool hasActiveChanged = previousIsActive != isActive();
    m_active = active;
    if (m_isInterrupted && m_active) {
        callOnMainThread([hasActiveChanged] {
            Ref session = sharedSession();
            if (session->m_isInterrupted && session->m_active)
                session->endInterruption(MayResume::Yes);
            if (hasActiveChanged)
                session->activeStateChanged();
        });
    } else if (hasActiveChanged)
        activeStateChanged();

    return true;
}

void AudioSession::addInterruptionObserver(AudioSessionInterruptionObserver& observer)
{
    m_interruptionObservers.add(observer);
}

void AudioSession::removeInterruptionObserver(AudioSessionInterruptionObserver& observer)
{
    m_interruptionObservers.remove(observer);
}

void AudioSession::beginInterruption()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    if (m_isInterrupted) {
        RELEASE_LOG_ERROR(WebRTC, "AudioSession::beginInterruption but session is already interrupted!");
        return;
    }
    m_isInterrupted = true;
    for (auto& observer : m_interruptionObservers)
        observer.beginAudioSessionInterruption();
}

void AudioSession::endInterruption(MayResume mayResume)
{
    ALWAYS_LOG(LOGIDENTIFIER);
    if (!m_isInterrupted) {
        RELEASE_LOG_ERROR(WebRTC, "AudioSession::endInterruption but session is already uninterrupted!");
        return;
    }
    m_isInterrupted = false;

    for (auto& observer : m_interruptionObservers)
        observer.endAudioSessionInterruption(mayResume);
}

void AudioSession::activeStateChanged()
{
    for (auto& observer : m_interruptionObservers)
        observer.audioSessionActiveStateChanged();
}

void AudioSession::setCategory(CategoryType, Mode, RouteSharingPolicy)
{
    notImplemented();
}

void AudioSession::setCategoryOverride(CategoryType category)
{
    if (m_categoryOverride == category)
        return;

    ALWAYS_LOG(LOGIDENTIFIER);

    m_categoryOverride = category;
    if (category != CategoryType::None)
        setCategory(category, Mode::Default, RouteSharingPolicy::Default);
}

AudioSession::CategoryType AudioSession::categoryOverride() const
{
    return m_categoryOverride;
}

AudioSession::CategoryType AudioSession::category() const
{
    notImplemented();
    return AudioSession::CategoryType::None;
}

AudioSession::Mode AudioSession::mode() const
{
    notImplemented();
    return AudioSession::Mode::Default;
}

float AudioSession::sampleRate() const
{
    notImplemented();
    return 0;
}

size_t AudioSession::bufferSize() const
{
    notImplemented();
    return 0;
}

size_t AudioSession::numberOfOutputChannels() const
{
    notImplemented();
    return 0;
}

size_t AudioSession::maximumNumberOfOutputChannels() const
{
    notImplemented();
    return 0;
}

bool AudioSession::tryToSetActiveInternal(bool)
{
    notImplemented();
    return true;
}

size_t AudioSession::preferredBufferSize() const
{
    notImplemented();
    return 0;
}

void AudioSession::setPreferredBufferSize(size_t)
{
    notImplemented();
}

RouteSharingPolicy AudioSession::routeSharingPolicy() const
{
    return RouteSharingPolicy::Default;
}

String AudioSession::routingContextUID() const
{
    return emptyString();
}

void AudioSession::audioOutputDeviceChanged()
{
    notImplemented();
}

void AudioSession::addConfigurationChangeObserver(AudioSessionConfigurationChangeObserver&)
{
    notImplemented();
}

void AudioSession::removeConfigurationChangeObserver(AudioSessionConfigurationChangeObserver&)
{
    notImplemented();
}

void AudioSession::setIsPlayingToBluetoothOverride(std::optional<bool>)
{
    notImplemented();
}

Logger& AudioSession::logger()
{
    if (!m_logger)
        m_logger = Logger::create(this);

    return *m_logger;
}

WTFLogChannel& AudioSession::logChannel() const
{
    return LogMedia;
}

String convertEnumerationToString(RouteSharingPolicy enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 4> values {
        MAKE_STATIC_STRING_IMPL("Default"),
        MAKE_STATIC_STRING_IMPL("LongFormAudio"),
        MAKE_STATIC_STRING_IMPL("Independent"),
        MAKE_STATIC_STRING_IMPL("LongFormVideo"),
    };
    static_assert(!static_cast<size_t>(RouteSharingPolicy::Default), "RouteSharingPolicy::Default is not 0 as expected");
    static_assert(static_cast<size_t>(RouteSharingPolicy::LongFormAudio) == 1, "RouteSharingPolicy::LongFormAudio is not 1 as expected");
    static_assert(static_cast<size_t>(RouteSharingPolicy::Independent) == 2, "RouteSharingPolicy::Independent is not 2 as expected");
    static_assert(static_cast<size_t>(RouteSharingPolicy::LongFormVideo) == 3, "RouteSharingPolicy::LongFormVideo is not 3 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

String convertEnumerationToString(AudioSession::CategoryType enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 7> values {
        MAKE_STATIC_STRING_IMPL("None"),
        MAKE_STATIC_STRING_IMPL("AmbientSound"),
        MAKE_STATIC_STRING_IMPL("SoloAmbientSound"),
        MAKE_STATIC_STRING_IMPL("MediaPlayback"),
        MAKE_STATIC_STRING_IMPL("RecordAudio"),
        MAKE_STATIC_STRING_IMPL("PlayAndRecord"),
        MAKE_STATIC_STRING_IMPL("AudioProcessing"),
    };
    static_assert(!static_cast<size_t>(AudioSession::CategoryType::None), "AudioSession::CategoryType::None is not 0 as expected");
    static_assert(static_cast<size_t>(AudioSession::CategoryType::AmbientSound) == 1, "AudioSession::CategoryType::AmbientSound is not 1 as expected");
    static_assert(static_cast<size_t>(AudioSession::CategoryType::SoloAmbientSound) == 2, "AudioSession::CategoryType::SoloAmbientSound is not 2 as expected");
    static_assert(static_cast<size_t>(AudioSession::CategoryType::MediaPlayback) == 3, "AudioSession::CategoryType::MediaPlayback is not 3 as expected");
    static_assert(static_cast<size_t>(AudioSession::CategoryType::RecordAudio) == 4, "AudioSession::CategoryType::RecordAudio is not 4 as expected");
    static_assert(static_cast<size_t>(AudioSession::CategoryType::PlayAndRecord) == 5, "AudioSession::CategoryType::PlayAndRecord is not 5 as expected");
    static_assert(static_cast<size_t>(AudioSession::CategoryType::AudioProcessing) == 6, "AudioSession::CategoryType::AudioProcessing is not 6 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

String convertEnumerationToString(AudioSession::Mode enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 3> values {
        MAKE_STATIC_STRING_IMPL("Default"),
        MAKE_STATIC_STRING_IMPL("VideoChat"),
        MAKE_STATIC_STRING_IMPL("MoviePlayback"),
    };
    static_assert(!static_cast<size_t>(AudioSession::Mode::Default), "AudioSession::Mode::Default is not 0 as expected");
    static_assert(static_cast<size_t>(AudioSession::Mode::VideoChat) == 1, "AudioSession::Mode::VideoChat is not 1 as expected");
    static_assert(static_cast<size_t>(AudioSession::Mode::MoviePlayback) == 2, "AudioSession::Mode::MoviePlayback is not 2 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

String convertEnumerationToString(AudioSessionRoutingArbitrationClient::RoutingArbitrationError enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 3> values {
        MAKE_STATIC_STRING_IMPL("None"),
        MAKE_STATIC_STRING_IMPL("Failed"),
        MAKE_STATIC_STRING_IMPL("Cancelled"),
    };
    static_assert(!static_cast<size_t>(AudioSessionRoutingArbitrationClient::RoutingArbitrationError::None), "AudioSessionRoutingArbitrationClient::RoutingArbitrationError::None is not 0 as expected");
    static_assert(static_cast<size_t>(AudioSessionRoutingArbitrationClient::RoutingArbitrationError::Failed), "AudioSessionRoutingArbitrationClient::RoutingArbitrationError::Failed is not 1 as expected");
    static_assert(static_cast<size_t>(AudioSessionRoutingArbitrationClient::RoutingArbitrationError::Cancelled), "AudioSessionRoutingArbitrationClient::RoutingArbitrationError::Cancelled is not 2 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

String convertEnumerationToString(AudioSessionRoutingArbitrationClient::DefaultRouteChanged enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 2> values {
        MAKE_STATIC_STRING_IMPL("No"),
        MAKE_STATIC_STRING_IMPL("Yes"),
    };
    static_assert(!static_cast<bool>(AudioSessionRoutingArbitrationClient::DefaultRouteChanged::No), "AudioSessionRoutingArbitrationClient::DefaultRouteChanged::No is not false as expected");
    static_assert(static_cast<bool>(AudioSessionRoutingArbitrationClient::DefaultRouteChanged::Yes), "AudioSessionRoutingArbitrationClient::DefaultRouteChanged::Yes is not true as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

String convertEnumerationToString(AudioSession::SoundStageSize size)
{
    static const std::array<NeverDestroyed<String>, 4> values {
        MAKE_STATIC_STRING_IMPL("Automatic"),
        MAKE_STATIC_STRING_IMPL("Small"),
        MAKE_STATIC_STRING_IMPL("Medium"),
        MAKE_STATIC_STRING_IMPL("Large"),
    };
    static_assert(!static_cast<size_t>(AudioSession::SoundStageSize::Automatic), "AudioSession::SoundStageSize::Automatic is not 0 as expected");
    static_assert(static_cast<size_t>(AudioSession::SoundStageSize::Small) == 1, "AudioSession::SoundStageSize::Small is not 1 as expected");
    static_assert(static_cast<size_t>(AudioSession::SoundStageSize::Medium) == 2, "AudioSession::SoundStageSize::Medium is not 2 as expected");
    static_assert(static_cast<size_t>(AudioSession::SoundStageSize::Large) == 3, "AudioSession::SoundStageSize::Large is not 3 as expected");
    ASSERT(static_cast<size_t>(size) < std::size(values));
    return values[static_cast<size_t>(size)];
}

}

#endif // USE(AUDIO_SESSION)
