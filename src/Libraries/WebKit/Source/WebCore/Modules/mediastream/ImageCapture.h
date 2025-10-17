/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include "Blob.h"
#include "Document.h"
#include "JSDOMPromiseDeferred.h"
#include "MediaStreamTrack.h"
#include "PhotoCapabilities.h"
#include "PhotoSettings.h"

namespace WTF {
class Logger;
}

namespace WebCore {

class ImageCapture : public RefCounted<ImageCapture>, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageCapture);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static ExceptionOr<Ref<ImageCapture>> create(Document&, Ref<MediaStreamTrack>);

    ~ImageCapture();

    void takePhoto(PhotoSettings&&, DOMPromiseDeferred<IDLInterface<Blob>>&&);
    void getPhotoCapabilities(DOMPromiseDeferred<IDLDictionary<PhotoCapabilities>>&&);
    void getPhotoSettings(DOMPromiseDeferred<IDLDictionary<PhotoSettings>>&&);

    Ref<MediaStreamTrack> track() const { return m_track; }

private:
    ImageCapture(Document&, Ref<MediaStreamTrack>);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger.get(); }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "ImageCapture"_s; }
    WTFLogChannel& logChannel() const;
#endif

    Ref<MediaStreamTrack> m_track;
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

}

#endif
