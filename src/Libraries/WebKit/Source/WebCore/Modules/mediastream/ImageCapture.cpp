/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
#include "ImageCapture.h"

#if ENABLE(MEDIA_STREAM)

#include "JSBlob.h"
#include "JSPhotoCapabilities.h"
#include "JSPhotoSettings.h"
#include "Logging.h"
#include "TaskSource.h"
#include <wtf/LoggerHelper.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageCapture);

ExceptionOr<Ref<ImageCapture>> ImageCapture::create(Document& document, Ref<MediaStreamTrack> track)
{
    if (track->kind() != "video"_s)
        return Exception { ExceptionCode::NotSupportedError, "Invalid track kind"_s };

    auto imageCapture = adoptRef(*new ImageCapture(document, track));
    imageCapture->suspendIfNeeded();
    return imageCapture;
}

ImageCapture::ImageCapture(Document& document, Ref<MediaStreamTrack> track)
    : ActiveDOMObject(document)
    , m_track(track)
#if !RELEASE_LOG_DISABLED
    , m_logger(track->logger())
    , m_logIdentifier(track->logIdentifier())
#endif
{
    ALWAYS_LOG(LOGIDENTIFIER);
}

ImageCapture::~ImageCapture() = default;

void ImageCapture::takePhoto(PhotoSettings&& settings, DOMPromiseDeferred<IDLInterface<Blob>>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    // https://w3c.github.io/mediacapture-image/#dom-imagecapture-takephoto
    // If the readyState of track provided in the constructor is not live, return
    // a promise rejected with a new DOMException whose name is InvalidStateError,
    // and abort these steps.
    if (m_track->readyState() == MediaStreamTrack::State::Ended) {
        ERROR_LOG(identifier, "rejecting promise, track has ended");
        promise.reject(Exception { ExceptionCode::InvalidStateError, "Track has ended"_s });
        return;
    }

    m_track->takePhoto(WTFMove(settings))->whenSettled(RunLoop::main(), [this, protectedThis = Ref { *this }, promise = WTFMove(promise), identifier = WTFMove(identifier)] (auto&& result) mutable {
        queueTaskKeepingObjectAlive(*this, TaskSource::ImageCapture, [this, promise = WTFMove(promise), result = WTFMove(result), identifier = WTFMove(identifier)] () mutable {
            if (!result) {
                ERROR_LOG(identifier, "rejecting promise: ", result.error().message());
                promise.reject(WTFMove(result.error()));
                return;
            }

            ALWAYS_LOG(identifier, "resolving promise");
            promise.resolve(Blob::create(scriptExecutionContext(), WTFMove(get<0>(result.value())), WTFMove(get<1>(result.value()))));
        });
    });
}

void ImageCapture::getPhotoCapabilities(DOMPromiseDeferred<IDLDictionary<PhotoCapabilities>>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    // https://w3c.github.io/mediacapture-image/#dom-imagecapture-getphotocapabilities
    // If the readyState of track provided in the constructor is not live, return
    // a promise rejected with a new DOMException whose name is InvalidStateError,
    // and abort these steps.
    if (m_track->readyState() == MediaStreamTrack::State::Ended) {
        ERROR_LOG(identifier, "rejecting promise, track has ended");
        promise.reject(Exception { ExceptionCode::InvalidStateError, "Track has ended"_s });
        return;
    }

    m_track->getPhotoCapabilities()->whenSettled(RunLoop::main(), [this, protectedThis = Ref { *this }, promise = WTFMove(promise), identifier = WTFMove(identifier)] (auto&& result) mutable {
        queueTaskKeepingObjectAlive(*this, TaskSource::ImageCapture, [this, promise = WTFMove(promise), result = WTFMove(result), identifier = WTFMove(identifier)] () mutable {
            if (!result) {
                ERROR_LOG(identifier, "rejecting promise: ", result.error().message());
                promise.reject(WTFMove(result.error()));
                return;
            }

            ALWAYS_LOG(identifier, "resolving promise");
            promise.resolve(WTFMove(result.value()));
        });
    });
}

void ImageCapture::getPhotoSettings(DOMPromiseDeferred<IDLDictionary<PhotoSettings>>&& promise)
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG(identifier);

    // https://w3c.github.io/mediacapture-image/#ref-for-dom-imagecapture-getphotosettingsâ‘¡
    // If the readyState of track provided in the constructor is not live, return
    // a promise rejected with a new DOMException whose name is InvalidStateError,
    // and abort these steps.
    if (m_track->readyState() == MediaStreamTrack::State::Ended) {
        ERROR_LOG(identifier, "rejecting promise, track has ended");
        promise.reject(Exception { ExceptionCode::InvalidStateError, "Track has ended"_s });
        return;
    }

    m_track->getPhotoSettings()->whenSettled(RunLoop::main(), [this, protectedThis = Ref { *this }, promise = WTFMove(promise), identifier = WTFMove(identifier)] (auto&& result) mutable {
        queueTaskKeepingObjectAlive(*this, TaskSource::ImageCapture, [this, promise = WTFMove(promise), result = WTFMove(result), identifier = WTFMove(identifier)] () mutable {
            if (!result) {
                ERROR_LOG(identifier, "rejecting promise: ", result.error().message());
                promise.reject(WTFMove(result.error()));
                return;
            }

            ALWAYS_LOG(identifier, "resolving promise");
            promise.resolve(WTFMove(result.value()));
        });
    });
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& ImageCapture::logChannel() const
{
    return LogWebRTC;
}
#endif

} // namespace WebCore

#endif
