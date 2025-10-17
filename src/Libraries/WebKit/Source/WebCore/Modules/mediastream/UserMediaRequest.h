/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "ActiveDOMObject.h"
#include "CaptureDevice.h"
#include "IDLTypes.h"
#include "MediaAccessDenialReason.h"
#include "MediaConstraints.h"
#include "MediaStreamPrivate.h"
#include "MediaStreamTrack.h"
#include "MediaStreamRequest.h"
#include "UserMediaRequestIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Identified.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class MediaStream;
class SecurityOrigin;

template<typename IDLType> class DOMPromiseDeferred;

class UserMediaRequest : public RefCounted<UserMediaRequest>, public ActiveDOMObject, public Identified<UserMediaRequestIdentifier> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using TrackConstraints = std::variant<bool, MediaTrackConstraints>;
    static Ref<UserMediaRequest> create(Document&, MediaStreamRequest&&, TrackConstraints&&, TrackConstraints&&, DOMPromiseDeferred<IDLInterface<MediaStream>>&&);
    virtual ~UserMediaRequest();

    void start();

    WEBCORE_EXPORT void setAllowedMediaDeviceUIDs(const String& audioDeviceUID, const String& videoDeviceUID);
    WEBCORE_EXPORT void allow(CaptureDevice&& audioDevice, CaptureDevice&& videoDevice, MediaDeviceHashSalts&&, CompletionHandler<void()>&&);

    WEBCORE_EXPORT void deny(MediaAccessDenialReason, const String& errorMessage = emptyString(), MediaConstraintType = MediaConstraintType::Unknown);

    const Vector<String>& audioDeviceUIDs() const { return m_audioDeviceUIDs; }
    const Vector<String>& videoDeviceUIDs() const { return m_videoDeviceUIDs; }

    const MediaConstraints& audioConstraints() const { return m_request.audioConstraints; }
    const MediaConstraints& videoConstraints() const { return m_request.videoConstraints; }

    WEBCORE_EXPORT SecurityOrigin* userMediaDocumentOrigin() const;
    WEBCORE_EXPORT SecurityOrigin* topLevelDocumentOrigin() const;
    WEBCORE_EXPORT Document* document() const;

    const MediaStreamRequest& request() const { return m_request; }

private:
    UserMediaRequest(Document&, MediaStreamRequest&&, TrackConstraints&&, TrackConstraints&&, DOMPromiseDeferred<IDLInterface<MediaStream>>&&);

    // ActiveDOMObject.
    void stop() final;

    Vector<String> m_videoDeviceUIDs;
    Vector<String> m_audioDeviceUIDs;

    UniqueRef<DOMPromiseDeferred<IDLInterface<MediaStream>>> m_promise;
    CompletionHandler<void()> m_allowCompletionHandler;
    MediaStreamRequest m_request;
    TrackConstraints m_audioConstraints;
    TrackConstraints m_videoConstraints;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
