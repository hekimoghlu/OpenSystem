/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#include "HTMLVideoElementPictureInPicture.h"

#if ENABLE(PICTURE_IN_PICTURE_API)

#include "EventNames.h"
#include "HTMLVideoElement.h"
#include "JSDOMPromiseDeferred.h"
#include "JSPictureInPictureWindow.h"
#include "Logging.h"
#include "PictureInPictureEvent.h"
#include "PictureInPictureSupport.h"
#include "PictureInPictureWindow.h"
#include "VideoTrackList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLVideoElementPictureInPicture);

HTMLVideoElementPictureInPicture::HTMLVideoElementPictureInPicture(HTMLVideoElement& videoElement)
    : m_videoElement(videoElement)
    , m_pictureInPictureWindow(PictureInPictureWindow::create(videoElement.document()))
#if !RELEASE_LOG_DISABLED
    , m_logger(videoElement.document().logger())
    , m_logIdentifier(uniqueLogIdentifier())
#endif
{
    ALWAYS_LOG(LOGIDENTIFIER);
    m_videoElement->setPictureInPictureObserver(this);
}

HTMLVideoElementPictureInPicture::~HTMLVideoElementPictureInPicture()
{
    ALWAYS_LOG(LOGIDENTIFIER);
    m_videoElement->setPictureInPictureObserver(nullptr);
}

HTMLVideoElementPictureInPicture* HTMLVideoElementPictureInPicture::from(HTMLVideoElement& videoElement)
{
    HTMLVideoElementPictureInPicture* supplement = static_cast<HTMLVideoElementPictureInPicture*>(Supplement<HTMLVideoElement>::from(&videoElement, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<HTMLVideoElementPictureInPicture>(videoElement);
        supplement = newSupplement.get();
        provideTo(&videoElement, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

void HTMLVideoElementPictureInPicture::providePictureInPictureTo(HTMLVideoElement& videoElement)
{
    provideTo(&videoElement, supplementName(), makeUnique<HTMLVideoElementPictureInPicture>(videoElement));
}

void HTMLVideoElementPictureInPicture::requestPictureInPicture(HTMLVideoElement& videoElement, Ref<DeferredPromise>&& promise)
{
    if (!supportsPictureInPicture()) {
        promise->reject(ExceptionCode::NotSupportedError, "The Picture-in-Picture mode is not supported."_s);
        return;
    }

    if (videoElement.readyState() == HTMLMediaElementEnums::HAVE_NOTHING) {
        promise->reject(ExceptionCode::InvalidStateError, "The video element is not ready to enter the Picture-in-Picture mode."_s);
        return;
    }

    if (!videoElement.videoTracks() || !videoElement.videoTracks()->length()) {
        promise->reject(ExceptionCode::InvalidStateError, "The video element does not have a video track or it has not detected a video track yet."_s);
        return;
    }

    bool userActivationRequired = !videoElement.document().pictureInPictureElement();
    if (userActivationRequired && !UserGestureIndicator::processingUserGesture()) {
        promise->reject(ExceptionCode::NotAllowedError, "The request is not triggered by a user activation."_s);
        return;
    }

    auto videoElementPictureInPicture = HTMLVideoElementPictureInPicture::from(videoElement);
    if (videoElement.document().pictureInPictureElement() == &videoElement) {
        promise->resolve<IDLInterface<PictureInPictureWindow>>(*(videoElementPictureInPicture->m_pictureInPictureWindow));
        return;
    }

    if (videoElementPictureInPicture->m_enterPictureInPicturePromise || videoElementPictureInPicture->m_exitPictureInPicturePromise) {
        promise->reject(ExceptionCode::NotAllowedError, "The video element is processing a Picture-in-Picture request."_s);
        return;
    }

    if (videoElement.webkitSupportsPresentationMode(HTMLVideoElement::VideoPresentationMode::PictureInPicture)) {
        videoElementPictureInPicture->m_enterPictureInPicturePromise = WTFMove(promise);
        videoElement.webkitSetPresentationMode(HTMLVideoElement::VideoPresentationMode::PictureInPicture);
    } else
        promise->reject(ExceptionCode::NotSupportedError, "The video element does not support the Picture-in-Picture mode."_s);
}

bool HTMLVideoElementPictureInPicture::autoPictureInPicture(HTMLVideoElement& videoElement)
{
    return HTMLVideoElementPictureInPicture::from(videoElement)->m_autoPictureInPicture;
}

void HTMLVideoElementPictureInPicture::setAutoPictureInPicture(HTMLVideoElement& videoElement, bool autoPictureInPicture)
{
    HTMLVideoElementPictureInPicture::from(videoElement)->m_autoPictureInPicture = autoPictureInPicture;
}

bool HTMLVideoElementPictureInPicture::disablePictureInPicture(HTMLVideoElement& videoElement)
{
    return HTMLVideoElementPictureInPicture::from(videoElement)->m_disablePictureInPicture;
}

void HTMLVideoElementPictureInPicture::setDisablePictureInPicture(HTMLVideoElement& videoElement, bool disablePictureInPicture)
{
    HTMLVideoElementPictureInPicture::from(videoElement)->m_disablePictureInPicture = disablePictureInPicture;
}

void HTMLVideoElementPictureInPicture::exitPictureInPicture(Ref<DeferredPromise>&& promise)
{
    INFO_LOG(LOGIDENTIFIER);
    if (m_enterPictureInPicturePromise || m_exitPictureInPicturePromise) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    m_exitPictureInPicturePromise = WTFMove(promise);
    m_videoElement->webkitSetPresentationMode(HTMLVideoElement::VideoPresentationMode::Inline);
}

void HTMLVideoElementPictureInPicture::didEnterPictureInPicture(const IntSize& windowSize)
{
    INFO_LOG(LOGIDENTIFIER);
    m_videoElement->document().setPictureInPictureElement(m_videoElement.ptr());
    m_pictureInPictureWindow->setSize(windowSize);

    PictureInPictureEvent::Init initializer;
    initializer.bubbles = true;
    initializer.pictureInPictureWindow = m_pictureInPictureWindow;
    m_videoElement->scheduleEvent(PictureInPictureEvent::create(eventNames().enterpictureinpictureEvent, WTFMove(initializer)));

    if (m_enterPictureInPicturePromise) {
        m_enterPictureInPicturePromise->resolve<IDLInterface<PictureInPictureWindow>>(*m_pictureInPictureWindow);
        m_enterPictureInPicturePromise = nullptr;
    }
}

void HTMLVideoElementPictureInPicture::didExitPictureInPicture()
{
    INFO_LOG(LOGIDENTIFIER);
    m_pictureInPictureWindow->close();
    m_videoElement->document().setPictureInPictureElement(nullptr);

    PictureInPictureEvent::Init initializer;
    initializer.bubbles = true;
    initializer.pictureInPictureWindow = m_pictureInPictureWindow;
    m_videoElement->scheduleEvent(PictureInPictureEvent::create(eventNames().leavepictureinpictureEvent, WTFMove(initializer)));

    if (m_exitPictureInPicturePromise) {
        m_exitPictureInPicturePromise->resolve();
        m_exitPictureInPicturePromise = nullptr;
    }
}

void HTMLVideoElementPictureInPicture::pictureInPictureWindowResized(const IntSize& windowSize)
{
    m_pictureInPictureWindow->setSize(windowSize);
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& HTMLVideoElementPictureInPicture::logChannel() const
{
    return LogMedia;
}
#endif

} // namespace WebCore

#endif // ENABLE(PICTURE_IN_PICTURE_API)
