/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include "UserMediaRequest.h"

#if ENABLE(MEDIA_STREAM)

#include "AudioSession.h"
#include "DocumentInlines.h"
#include "JSDOMPromiseDeferred.h"
#include "JSMediaStream.h"
#include "JSOverconstrainedError.h"
#include "LocalFrame.h"
#include "Logging.h"
#include "MediaConstraints.h"
#include "MediaDevices.h"
#include "Navigator.h"
#include "NavigatorMediaDevices.h"
#include "PermissionsPolicy.h"
#include "PlatformMediaSessionManager.h"
#include "RTCController.h"
#include "RealtimeMediaSourceCenter.h"
#include "Settings.h"
#include "UserMediaController.h"
#include "WindowEventLoop.h"
#include <wtf/Scope.h>

namespace WebCore {

Ref<UserMediaRequest> UserMediaRequest::create(Document& document, MediaStreamRequest&& request, TrackConstraints&& audioConstraints, TrackConstraints&& videoConstraints, DOMPromiseDeferred<IDLInterface<MediaStream>>&& promise)
{
    auto result = adoptRef(*new UserMediaRequest(document, WTFMove(request), WTFMove(audioConstraints), WTFMove(videoConstraints), WTFMove(promise)));
    result->suspendIfNeeded();
    return result;
}

UserMediaRequest::UserMediaRequest(Document& document, MediaStreamRequest&& request, TrackConstraints&& audioConstraints, TrackConstraints&& videoConstraints, DOMPromiseDeferred<IDLInterface<MediaStream>>&& promise)
    : ActiveDOMObject(document)
    , m_promise(makeUniqueRef<DOMPromiseDeferred<IDLInterface<MediaStream>>>(WTFMove(promise)))
    , m_request(WTFMove(request))
    , m_audioConstraints(WTFMove(audioConstraints))
    , m_videoConstraints(WTFMove(videoConstraints))
{
}

UserMediaRequest::~UserMediaRequest()
{
    if (auto completionHandler = std::exchange(m_allowCompletionHandler, { }))
        completionHandler();
}

SecurityOrigin* UserMediaRequest::userMediaDocumentOrigin() const
{
    RefPtr context = scriptExecutionContext();
    return context ? context->securityOrigin() : nullptr;
}

SecurityOrigin* UserMediaRequest::topLevelDocumentOrigin() const
{
    RefPtr context = scriptExecutionContext();
    return context ? &context->topOrigin() : nullptr;
}

void UserMediaRequest::start()
{
    RefPtr context = scriptExecutionContext();
    ASSERT(context);
    if (!context) {
        deny(MediaAccessDenialReason::UserMediaDisabled);
        return;
    }

    // 4. If the current settings object's responsible document is NOT allowed to use the feature indicated by
    //    attribute name allowusermedia, return a promise rejected with a DOMException object whose name
    //    attribute has the value SecurityError.
    auto& document = downcast<Document>(*context);
    auto* controller = UserMediaController::from(document.page());
    if (!controller) {
        deny(MediaAccessDenialReason::UserMediaDisabled);
        return;
    }

    // 6.3 Optionally, e.g., based on a previously-established user preference, for security reasons,
    //     or due to platform limitations, jump to the step labeled Permission Failure below.
    // ...
    // 6.10 Permission Failure: Reject p with a new DOMException object whose name attribute has
    //      the value NotAllowedError.

    switch (m_request.type) {
    case MediaStreamRequest::Type::DisplayMedia:
    case MediaStreamRequest::Type::DisplayMediaWithAudio:
        if (!PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::DisplayCapture, document)) {
            deny(MediaAccessDenialReason::PermissionDenied);
            controller->logGetDisplayMediaDenial(document);
            return;
        }
        break;
    case MediaStreamRequest::Type::UserMedia:
        if (m_request.audioConstraints.isValid && !PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, document)) {
            deny(MediaAccessDenialReason::PermissionDenied);
            controller->logGetUserMediaDenial(document);
            return;
        }
        if (m_request.videoConstraints.isValid && !PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Camera, document)) {
            deny(MediaAccessDenialReason::PermissionDenied);
            controller->logGetUserMediaDenial(document);
            return;
        }
        break;
    }

    ASSERT(document.page());
    if (RefPtr page = document.protectedPage())
        PlatformMediaSessionManager::singleton().prepareToSendUserMediaPermissionRequestForPage(*page);
    controller->requestUserMediaAccess(*this);
}

static inline bool isMediaStreamCorrectlyStarted(const MediaStream& stream)
{
    if (stream.getTracks().isEmpty())
        return false;

    return WTF::allOf(stream.getTracks(), [](auto& track) {
        return !track->source().captureDidFail();
    });
}

void UserMediaRequest::allow(CaptureDevice&& audioDevice, CaptureDevice&& videoDevice, MediaDeviceHashSalts&& deviceIdentifierHashSalt, CompletionHandler<void()>&& completionHandler)
{
    RELEASE_LOG(MediaStream, "UserMediaRequest::allow %s %s", audioDevice ? audioDevice.persistentId().utf8().data() : "", videoDevice ? videoDevice.persistentId().utf8().data() : "");

    Ref document = downcast<Document>(*scriptExecutionContext());
    RefPtr localWindow = document->protectedWindow();
    RefPtr mediaDevices = localWindow ? NavigatorMediaDevices::mediaDevices(localWindow->protectedNavigator()) : nullptr;
    if (mediaDevices)
        mediaDevices->willStartMediaCapture(!!audioDevice, !!videoDevice);

    m_allowCompletionHandler = WTFMove(completionHandler);
    queueTaskKeepingObjectAlive(*this, TaskSource::UserInteraction, [this, audioDevice = WTFMove(audioDevice), videoDevice = WTFMove(videoDevice), deviceIdentifierHashSalt = WTFMove(deviceIdentifierHashSalt)]() mutable {
        auto callback = [this, protector = makePendingActivity(*this)](auto privateStreamOrError) mutable {
            auto scopeExit = makeScopeExit([completionHandler = WTFMove(m_allowCompletionHandler)]() mutable {
                completionHandler();
            });
            if (isContextStopped()) {
                if (!!privateStreamOrError) {
                    RELEASE_LOG(MediaStream, "UserMediaRequest::allow, context is stopped");
                    privateStreamOrError.value()->forEachTrack([](auto& track) {
                        track.endTrack();
                    });
                }
                return;
            }

            if (!privateStreamOrError) {
                RELEASE_LOG(MediaStream, "UserMediaRequest::allow failed to create media stream!");
                auto error = privateStreamOrError.error();
                scriptExecutionContext()->addConsoleMessage(MessageSource::JS, MessageLevel::Error, error.errorMessage);
                deny(error.denialReason, error.errorMessage, error.invalidConstraint);
                return;
            }
            auto privateStream = WTFMove(privateStreamOrError).value();

            auto& document = downcast<Document>(*scriptExecutionContext());
            privateStream->monitorOrientation(document.orientationNotifier());

            auto stream = MediaStream::create(document, WTFMove(privateStream));
            stream->startProducingData();

            if (!isMediaStreamCorrectlyStarted(stream)) {
                deny(MediaAccessDenialReason::HardwareError);
                return;
            }

            if (RefPtr audioTrack = stream->getFirstAudioTrack()) {
#if USE(AUDIO_SESSION)
                AudioSession::sharedSession().tryToSetActive(true);
#endif
                if (std::holds_alternative<MediaTrackConstraints>(m_audioConstraints))
                    audioTrack->setConstraints(std::get<MediaTrackConstraints>(WTFMove(m_audioConstraints)));
            }
            if (RefPtr videoTrack = stream->getFirstVideoTrack()) {
                if (std::holds_alternative<MediaTrackConstraints>(m_videoConstraints))
                    videoTrack->setConstraints(std::get<MediaTrackConstraints>(WTFMove(m_videoConstraints)));
            }

            ASSERT(document.isCapturing());
            document.setHasCaptureMediaStreamTrack();
            m_promise->resolve(WTFMove(stream));
        };

        auto& document = downcast<Document>(*scriptExecutionContext());
        RealtimeMediaSourceCenter::singleton().createMediaStream(document.logger(), WTFMove(callback), WTFMove(deviceIdentifierHashSalt), WTFMove(audioDevice), WTFMove(videoDevice), m_request);

        if (!scriptExecutionContext())
            return;

#if ENABLE(WEB_RTC)
        if (auto* page = document.page())
            page->rtcController().disableICECandidateFilteringForDocument(document);
#endif
    });
}

void UserMediaRequest::deny(MediaAccessDenialReason reason, const String& message, MediaConstraintType invalidConstraint)
{
    if (!scriptExecutionContext())
        return;

    ExceptionCode code;
    switch (reason) {
    case MediaAccessDenialReason::NoReason:
        ASSERT_NOT_REACHED();
        code = ExceptionCode::AbortError;
        break;
    case MediaAccessDenialReason::NoConstraints:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - no constraints");
        code = ExceptionCode::TypeError;
        break;
    case MediaAccessDenialReason::UserMediaDisabled:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - user media disabled");
        code = ExceptionCode::SecurityError;
        break;
    case MediaAccessDenialReason::NoCaptureDevices:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - no capture devices");
        code = ExceptionCode::NotFoundError;
        break;
    case MediaAccessDenialReason::InvalidConstraint:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - invalid constraint - %d", (int)invalidConstraint);
        m_promise->rejectType<IDLInterface<OverconstrainedError>>(OverconstrainedError::create(invalidConstraint, "Invalid constraint"_s).get());
        return;
    case MediaAccessDenialReason::HardwareError:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - hardware error");
        code = ExceptionCode::NotReadableError;
        break;
    case MediaAccessDenialReason::OtherFailure:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - other failure");
        code = ExceptionCode::AbortError;
        break;
    case MediaAccessDenialReason::PermissionDenied:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - permission denied");
        code = ExceptionCode::NotAllowedError;
        break;
    case MediaAccessDenialReason::InvalidAccess:
        RELEASE_LOG(MediaStream, "UserMediaRequest::deny - invalid access");
        code = ExceptionCode::InvalidAccessError;
        break;
    }

    if (!message.isEmpty())
        m_promise->reject(code, message);
    else
        m_promise->reject(code);
}

void UserMediaRequest::stop()
{
    Ref document = downcast<Document>(*scriptExecutionContext());
    if (auto* controller = UserMediaController::from(document->page()))
        controller->cancelUserMediaAccessRequest(*this);
}

Document* UserMediaRequest::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
