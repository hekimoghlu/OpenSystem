/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "RealtimeMediaSourceCenter.h"

#if ENABLE(MEDIA_STREAM)

#include "DisplayCaptureManager.h"
#include "Logging.h"
#include "MediaDeviceHashSalts.h"
#include "MediaStreamPrivate.h"
#include <wtf/CallbackAggregator.h>
#include <wtf/HexNumber.h>
#include <wtf/SHA1.h>

namespace WebCore {

#if !USE(GSTREAMER)
static const Seconds deviceChangeDebounceTimerInterval { 200_ms };
#endif

RealtimeMediaSourceCenter& RealtimeMediaSourceCenter::singleton()
{
    ASSERT(isMainThread());
    static NeverDestroyed<RealtimeMediaSourceCenter> center;
    return center;
}

RealtimeMediaSourceCenter::RealtimeMediaSourceCenter()
    : m_debounceTimer(RunLoop::main(), this, &RealtimeMediaSourceCenter::triggerDevicesChangedObservers)
{
}

RealtimeMediaSourceCenter::~RealtimeMediaSourceCenter() = default;

RealtimeMediaSourceCenterObserver::~RealtimeMediaSourceCenterObserver() = default;

void RealtimeMediaSourceCenter::createMediaStream(Ref<const Logger>&& logger, NewMediaStreamHandler&& completionHandler, MediaDeviceHashSalts&& hashSalts, CaptureDevice&& audioDevice, CaptureDevice&& videoDevice, const MediaStreamRequest& request)
{
    Vector<Ref<RealtimeMediaSource>> audioSources;
    Vector<Ref<RealtimeMediaSource>> videoSources;

    RefPtr<RealtimeMediaSource> audioSource;
    if (audioDevice) {
        auto source = audioCaptureFactory().createAudioCaptureSource(WTFMove(audioDevice), MediaDeviceHashSalts { hashSalts }, &request.audioConstraints, request.pageIdentifier);
        if (!source) {
            completionHandler(makeUnexpected(WTFMove(source.error)));
            return;
        }
        audioSource = source.source();
    }

    RefPtr<RealtimeMediaSource> videoSource;
    if (videoDevice) {
        CaptureSourceOrError source;
        if (videoDevice.type() == CaptureDevice::DeviceType::Camera)
            source = videoCaptureFactory().createVideoCaptureSource(WTFMove(videoDevice), WTFMove(hashSalts), &request.videoConstraints, request.pageIdentifier);
        else
            source = displayCaptureFactory().createDisplayCaptureSource(WTFMove(videoDevice), WTFMove(hashSalts), &request.videoConstraints, request.pageIdentifier);

        if (!source) {
            completionHandler(makeUnexpected(WTFMove(source.error)));
            return;
        }
        videoSource = source.source();
    }

    CompletionHandler<void(CaptureSourceError&&)> whenAudioSourceReady = [audioSource, videoSource = WTFMove(videoSource), logger = WTFMove(logger), completionHandler = WTFMove(completionHandler)](auto&& error) mutable {
        if (error)
            return completionHandler(makeUnexpected(error));
        if (!videoSource)
            return completionHandler(MediaStreamPrivate::create(WTFMove(logger), WTFMove(audioSource), WTFMove(videoSource)));

        CompletionHandler<void(CaptureSourceError&&)> whenVideoSourceReady = [audioSource = WTFMove(audioSource), videoSource, logger = WTFMove(logger), completionHandler = WTFMove(completionHandler)](auto&& error) mutable {
            if (error)
                return completionHandler(makeUnexpected(error));
            completionHandler(MediaStreamPrivate::create(WTFMove(logger), WTFMove(audioSource), WTFMove(videoSource)));
        };
        videoSource->whenReady(WTFMove(whenVideoSourceReady));
    };
    if (!audioSource)
        return whenAudioSourceReady({ });
    audioSource->whenReady(WTFMove(whenAudioSourceReady));
}

void RealtimeMediaSourceCenter::getMediaStreamDevices(CompletionHandler<void(Vector<CaptureDevice>&&)>&& completion)
{
    auto shouldEnumerateDisplay = displayCaptureFactory().displayCaptureDeviceManager().requiresCaptureDevicesEnumeration();
    enumerateDevices(true, shouldEnumerateDisplay, true, true, [this, completion = WTFMove(completion)]() mutable {
        Vector<CaptureDevice> results;

        results.appendVector(audioCaptureFactory().audioCaptureDeviceManager().captureDevices());
        results.appendVector(videoCaptureFactory().videoCaptureDeviceManager().captureDevices());
        results.appendVector(audioCaptureFactory().speakerDevices());

        auto& displayCaptureDeviceManager = displayCaptureFactory().displayCaptureDeviceManager();
        if (displayCaptureDeviceManager.requiresCaptureDevicesEnumeration())
            results.appendVector(displayCaptureDeviceManager.captureDevices());

        completion(WTFMove(results));
    });
}

std::optional<RealtimeMediaSourceCapabilities> RealtimeMediaSourceCenter::getCapabilities(const CaptureDevice& device)
{
    if (device.type() == CaptureDevice::DeviceType::Camera) {
        auto source = videoCaptureFactory().createVideoCaptureSource({ device },  { "fake"_s, "fake"_s }, nullptr, std::nullopt);
        if (!source)
            return std::nullopt;
        return source.source()->capabilities();
    }
    if (device.type() == CaptureDevice::DeviceType::Microphone) {
        auto source = audioCaptureFactory().createAudioCaptureSource({ device }, { "fake"_s, "fake"_s }, nullptr, std::nullopt);
        if (!source)
            return std::nullopt;
        return source.source()->capabilities();
    }

    return std::nullopt;
}

static void addStringToSHA1(SHA1& sha1, const String& string)
{
    if (string.isEmpty())
        return;

    sha1.addUTF8Bytes(string);
}

String RealtimeMediaSourceCenter::hashStringWithSalt(const String& id, const String& hashSalt)
{
    if (id.isEmpty() || hashSalt.isEmpty())
        return emptyString();

    SHA1 sha1;

    addStringToSHA1(sha1, id);
    addStringToSHA1(sha1, hashSalt);
    
    SHA1::Digest digest;
    sha1.computeHash(digest);
    
    return String::fromLatin1(SHA1::hexDigest(digest).data());
}

void RealtimeMediaSourceCenter::addDevicesChangedObserver(RealtimeMediaSourceCenterObserver& observer)
{
    ASSERT(isMainThread());
    m_observers.add(observer);
}

void RealtimeMediaSourceCenter::removeDevicesChangedObserver(RealtimeMediaSourceCenterObserver& observer)
{
    ASSERT(isMainThread());
    m_observers.remove(observer);
}

void RealtimeMediaSourceCenter::captureDevicesChanged()
{
    ASSERT(isMainThread());

#if USE(GSTREAMER)
    triggerDevicesChangedObservers();
#else
    // When a device with camera and microphone is attached or detached, the CaptureDevice notification for
    // the different devices won't arrive at the same time so delay a bit so we can coalesce the callbacks.
    if (!m_debounceTimer.isActive())
        m_debounceTimer.startOneShot(deviceChangeDebounceTimerInterval);
#endif
}

void RealtimeMediaSourceCenter::captureDeviceWillBeRemoved(const String& persistentId)
{
    Ref protectedThis { *this };
    m_observers.forEach([&](Ref<RealtimeMediaSourceCenterObserver> observer) {
        observer->deviceWillBeRemoved(persistentId);
    });
}

void RealtimeMediaSourceCenter::triggerDevicesChangedObservers()
{
    Ref protectedThis { *this };
    m_observers.forEach([](Ref<RealtimeMediaSourceCenterObserver> observer) {
        observer->devicesChanged();
    });
}

void RealtimeMediaSourceCenter::getDisplayMediaDevices(const MediaStreamRequest& request, MediaDeviceHashSalts&& hashSalts, Vector<DeviceInfo>& displayDeviceInfo, MediaConstraintType& firstInvalidConstraint)
{
    if (!request.videoConstraints.isValid)
        return;

    for (auto& device : displayCaptureFactory().displayCaptureDeviceManager().captureDevices()) {
        if (!device.enabled())
            continue;

        auto sourceOrError = displayCaptureFactory().createDisplayCaptureSource(device, MediaDeviceHashSalts { hashSalts }, &request.videoConstraints, request.pageIdentifier);
        if (!sourceOrError)
            continue;

        if (auto invalidConstraint = sourceOrError.captureSource->hasAnyInvalidConstraint(request.videoConstraints)) {
            if (firstInvalidConstraint == MediaConstraintType::Unknown)
                firstInvalidConstraint = *invalidConstraint;
            continue;
        }

        displayDeviceInfo.append({ sourceOrError.captureSource->fitnessScore(), device });
    }
}

void RealtimeMediaSourceCenter::getUserMediaDevices(const MediaStreamRequest& request, MediaDeviceHashSalts&& hashSalts, Vector<DeviceInfo>& audioDeviceInfo, Vector<DeviceInfo>& videoDeviceInfo, MediaConstraintType& firstInvalidConstraint)
{
    if (request.audioConstraints.isValid) {
        bool sameFitnessScore = true;
        std::optional<double> fitnessScore;
        for (auto& device : audioCaptureFactory().audioCaptureDeviceManager().captureDevices()) {
            if (!device.enabled())
                continue;

            auto sourceOrError = audioCaptureFactory().createAudioCaptureSource(device, MediaDeviceHashSalts { hashSalts }, { }, request.pageIdentifier);
            if (!sourceOrError)
                continue;

            if (auto invalidConstraint = sourceOrError.captureSource->hasAnyInvalidConstraint(request.audioConstraints)) {
                if (firstInvalidConstraint == MediaConstraintType::Unknown)
                    firstInvalidConstraint = *invalidConstraint;
                continue;
            }

            if (sameFitnessScore) {
                if (!fitnessScore)
                    fitnessScore = sourceOrError.captureSource->fitnessScore();
                else
                    sameFitnessScore = *fitnessScore == sourceOrError.captureSource->fitnessScore();
            }
            audioDeviceInfo.append({ sourceOrError.captureSource->fitnessScore(), device });
        }

        // We mark the device as default if no constraint was applied to selecting the device.
        // This gives the capture process or the OS the freedom to select the best microphone.
        if (!audioDeviceInfo.isEmpty())
            audioDeviceInfo[0].device.setIsDefault(sameFitnessScore && firstInvalidConstraint == MediaConstraintType::Unknown);
    }

    if (request.videoConstraints.isValid) {
        for (auto& device : videoCaptureFactory().videoCaptureDeviceManager().captureDevices()) {
            if (!device.enabled())
                continue;

            auto sourceOrError = videoCaptureFactory().createVideoCaptureSource(device, MediaDeviceHashSalts { hashSalts }, { }, request.pageIdentifier);
            if (!sourceOrError)
                continue;

            if (auto invalidConstraint = sourceOrError.captureSource->hasAnyInvalidConstraint(request.videoConstraints)) {
                if (firstInvalidConstraint == MediaConstraintType::Unknown)
                    firstInvalidConstraint = *invalidConstraint;
                continue;
            }

            videoDeviceInfo.append({ sourceOrError.captureSource->fitnessScore(), device });
        }
    }
}

void RealtimeMediaSourceCenter::enumerateDevices(bool shouldEnumerateCamera, bool shouldEnumerateDisplay, bool shouldEnumerateMicrophone, bool shouldEnumerateSpeakers, CompletionHandler<void()>&& callback)
{
    auto callbackAggregator = CallbackAggregator::create(WTFMove(callback));
    if (shouldEnumerateCamera)
        videoCaptureFactory().videoCaptureDeviceManager().computeCaptureDevices([callbackAggregator] { });
    if (shouldEnumerateDisplay)
        displayCaptureFactory().displayCaptureDeviceManager().computeCaptureDevices([callbackAggregator] { });
    if (shouldEnumerateMicrophone)
        audioCaptureFactory().audioCaptureDeviceManager().computeCaptureDevices([callbackAggregator] { });
    if (shouldEnumerateSpeakers)
        audioCaptureFactory().computeSpeakerDevices([callbackAggregator] { });
}

void RealtimeMediaSourceCenter::validateRequestConstraints(ValidConstraintsHandler&& validHandler, InvalidConstraintsHandler&& invalidHandler, const MediaStreamRequest& request, MediaDeviceHashSalts&& deviceIdentifierHashSalts)
{
    bool shouldEnumerateCamera = request.videoConstraints.isValid;
    bool shouldEnumerateDisplay = displayCaptureFactory().displayCaptureDeviceManager().requiresCaptureDevicesEnumeration();
    bool shouldEnumerateMicrophone = request.audioConstraints.isValid;
    bool shouldEnumerateSpeakers = false;
    enumerateDevices(shouldEnumerateCamera, shouldEnumerateDisplay, shouldEnumerateMicrophone, shouldEnumerateSpeakers, [this, validHandler = WTFMove(validHandler), invalidHandler = WTFMove(invalidHandler), request, deviceIdentifierHashSalts = WTFMove(deviceIdentifierHashSalts)]() mutable {
        validateRequestConstraintsAfterEnumeration(WTFMove(validHandler), WTFMove(invalidHandler), request, WTFMove(deviceIdentifierHashSalts));
    });
}

void RealtimeMediaSourceCenter::validateRequestConstraintsAfterEnumeration(ValidConstraintsHandler&& validHandler, InvalidConstraintsHandler&& invalidHandler, const MediaStreamRequest& request, MediaDeviceHashSalts&& deviceIdentifierHashSalts)
{
    ASSERT(request.type != MediaStreamRequest::Type::DisplayMedia || request.type != MediaStreamRequest::Type::DisplayMediaWithAudio);
    struct {
        bool operator()(const DeviceInfo& a, const DeviceInfo& b)
        {
            return a.fitnessScore > b.fitnessScore;
        }
    } sortBasedOnFitnessScore;

    Vector<DeviceInfo> audioDeviceInfo;
    Vector<DeviceInfo> videoDeviceInfo;
    MediaConstraintType firstInvalidConstraint = MediaConstraintType::Unknown;

    auto& displayCaptureManager = displayCaptureFactory().displayCaptureDeviceManager();
    if (displayCaptureManager.requiresCaptureDevicesEnumeration() && (request.type == MediaStreamRequest::Type::DisplayMedia || request.type == MediaStreamRequest::Type::DisplayMediaWithAudio))
        getDisplayMediaDevices(request, MediaDeviceHashSalts { deviceIdentifierHashSalts }, videoDeviceInfo, firstInvalidConstraint);
    else
        getUserMediaDevices(request, MediaDeviceHashSalts { deviceIdentifierHashSalts }, audioDeviceInfo, videoDeviceInfo, firstInvalidConstraint);

    if (request.audioConstraints.isValid && audioDeviceInfo.isEmpty()) {
        WTFLogAlways("Audio capture was requested but no device was found amongst %zu devices", audioCaptureFactory().audioCaptureDeviceManager().captureDevices().size());
        request.audioConstraints.mandatoryConstraints.forEach([](auto constraintType, auto& constraint) { constraint.log(constraintType); });

        invalidHandler(firstInvalidConstraint);
        return;
    }

    if (request.videoConstraints.isValid && videoDeviceInfo.isEmpty()) {
        WTFLogAlways("Video capture was requested but no device was found amongst %zu devices", videoCaptureFactory().videoCaptureDeviceManager().captureDevices().size());
        request.videoConstraints.mandatoryConstraints.forEach([](auto constraintType, auto& constraint) { constraint.log(constraintType); });

        invalidHandler(firstInvalidConstraint);
        return;
    }

    Vector<CaptureDevice> audioDevices;
    if (!audioDeviceInfo.isEmpty()) {
        std::stable_sort(audioDeviceInfo.begin(), audioDeviceInfo.end(), sortBasedOnFitnessScore);
        audioDevices = WTF::map(audioDeviceInfo, [] (auto& info) {
            return info.device;
        });
    }

    Vector<CaptureDevice> videoDevices;
    if (!videoDeviceInfo.isEmpty()) {
        std::stable_sort(videoDeviceInfo.begin(), videoDeviceInfo.end(), sortBasedOnFitnessScore);
        videoDevices = WTF::map(videoDeviceInfo, [] (auto& info) {
            return info.device;
        });
    }

    validHandler(WTFMove(audioDevices), WTFMove(videoDevices));
}

void RealtimeMediaSourceCenter::setAudioCaptureFactory(AudioCaptureFactory& factory)
{
    m_audioCaptureFactoryOverride = &factory;
}

void RealtimeMediaSourceCenter::unsetAudioCaptureFactory(AudioCaptureFactory& oldOverride)
{
    ASSERT_UNUSED(oldOverride, m_audioCaptureFactoryOverride == &oldOverride);
    if (&oldOverride == m_audioCaptureFactoryOverride)
        m_audioCaptureFactoryOverride = nullptr;
}

AudioCaptureFactory& RealtimeMediaSourceCenter::audioCaptureFactory()
{
    return m_audioCaptureFactoryOverride ? *m_audioCaptureFactoryOverride : defaultAudioCaptureFactory();
}

void RealtimeMediaSourceCenter::setVideoCaptureFactory(VideoCaptureFactory& factory)
{
    m_videoCaptureFactoryOverride = &factory;
}
void RealtimeMediaSourceCenter::unsetVideoCaptureFactory(VideoCaptureFactory& oldOverride)
{
    ASSERT_UNUSED(oldOverride, m_videoCaptureFactoryOverride == &oldOverride);
    if (&oldOverride == m_videoCaptureFactoryOverride)
        m_videoCaptureFactoryOverride = nullptr;
}

VideoCaptureFactory& RealtimeMediaSourceCenter::videoCaptureFactory()
{
    return m_videoCaptureFactoryOverride ? *m_videoCaptureFactoryOverride : defaultVideoCaptureFactory();
}

void RealtimeMediaSourceCenter::setDisplayCaptureFactory(DisplayCaptureFactory& factory)
{
    m_displayCaptureFactoryOverride = &factory;
}

void RealtimeMediaSourceCenter::unsetDisplayCaptureFactory(DisplayCaptureFactory& oldOverride)
{
    ASSERT_UNUSED(oldOverride, m_displayCaptureFactoryOverride == &oldOverride);
    if (&oldOverride == m_displayCaptureFactoryOverride)
        m_displayCaptureFactoryOverride = nullptr;
}

DisplayCaptureFactory& RealtimeMediaSourceCenter::displayCaptureFactory()
{
    return m_displayCaptureFactoryOverride ? *m_displayCaptureFactoryOverride : defaultDisplayCaptureFactory();
}

#if !PLATFORM(COCOA)
bool RealtimeMediaSourceCenter::shouldInterruptAudioOnPageVisibilityChange()
{
    return false;
}
#endif

#if ENABLE(EXTENSION_CAPABILITIES)
const String& RealtimeMediaSourceCenter::currentMediaEnvironment() const
{
    return m_currentMediaEnvironment;
}

void RealtimeMediaSourceCenter::setCurrentMediaEnvironment(String&& mediaEnvironment)
{
    m_currentMediaEnvironment = WTFMove(mediaEnvironment);
}
#endif

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
