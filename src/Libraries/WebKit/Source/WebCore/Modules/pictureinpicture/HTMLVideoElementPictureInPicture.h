/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

#if ENABLE(PICTURE_IN_PICTURE_API)

#include "PictureInPictureObserver.h"
#include "Supplementable.h"
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class DeferredPromise;
class HTMLVideoElement;
class PictureInPictureWindow;

class HTMLVideoElementPictureInPicture
    : public Supplement<HTMLVideoElement>
    , public PictureInPictureObserver
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLVideoElementPictureInPicture);
public:
    HTMLVideoElementPictureInPicture(HTMLVideoElement&);
    static HTMLVideoElementPictureInPicture* from(HTMLVideoElement&);
    static void providePictureInPictureTo(HTMLVideoElement&);
    virtual ~HTMLVideoElementPictureInPicture();

    static void requestPictureInPicture(HTMLVideoElement&, Ref<DeferredPromise>&&);
    static bool autoPictureInPicture(HTMLVideoElement&);
    static void setAutoPictureInPicture(HTMLVideoElement&, bool);
    static bool disablePictureInPicture(HTMLVideoElement&);
    static void setDisablePictureInPicture(HTMLVideoElement&, bool);

    void exitPictureInPicture(Ref<DeferredPromise>&&);

    void didEnterPictureInPicture(const IntSize&);
    void didExitPictureInPicture();
    void pictureInPictureWindowResized(const IntSize&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "HTMLVideoElementPictureInPicture"_s; }
    WTFLogChannel& logChannel() const final;
#endif

private:
    static ASCIILiteral supplementName() { return "HTMLVideoElementPictureInPicture"_s; }

    bool m_autoPictureInPicture { false };
    bool m_disablePictureInPicture { false };

    WeakRef<HTMLVideoElement> m_videoElement;
    RefPtr<PictureInPictureWindow> m_pictureInPictureWindow;
    RefPtr<DeferredPromise> m_enterPictureInPicturePromise;
    RefPtr<DeferredPromise> m_exitPictureInPicturePromise;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebCore

#endif // ENABLE(PICTURE_IN_PICTURE_API)
