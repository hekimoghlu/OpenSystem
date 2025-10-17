/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

#include "APIObject.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/MediaKeySystemRequest.h>
#include <WebCore/MediaKeySystemRequestIdentifier.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

class MediaKeySystemPermissionRequestManagerProxy;

class MediaKeySystemPermissionRequestProxy : public RefCounted<MediaKeySystemPermissionRequestProxy> {
public:
    static Ref<MediaKeySystemPermissionRequestProxy> create(MediaKeySystemPermissionRequestManagerProxy& manager, WebCore::MediaKeySystemRequestIdentifier mediaKeySystemID, WebCore::FrameIdentifier mainFrameID, WebCore::FrameIdentifier frameID, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin, const String& keySystem)
    {
        return adoptRef(*new MediaKeySystemPermissionRequestProxy(manager, mediaKeySystemID, mainFrameID, frameID, WTFMove(topLevelDocumentOrigin), keySystem));
    }

    void allow();
    void deny();

    void invalidate();

    WebCore::MediaKeySystemRequestIdentifier mediaKeySystemID() const { return m_mediaKeySystemID; }
    WebCore::FrameIdentifier mainFrameID() const { return m_mainFrameID; }
    WebCore::FrameIdentifier frameID() const { return m_frameID; }

    WebCore::SecurityOrigin& topLevelDocumentSecurityOrigin() { return m_topLevelDocumentSecurityOrigin.get(); }
    const WebCore::SecurityOrigin& topLevelDocumentSecurityOrigin() const { return m_topLevelDocumentSecurityOrigin.get(); }

    const String& keySystem() const { return m_keySystem; }

    void doDefaultAction();

private:
    MediaKeySystemPermissionRequestProxy(MediaKeySystemPermissionRequestManagerProxy&, WebCore::MediaKeySystemRequestIdentifier, WebCore::FrameIdentifier mainFrameID, WebCore::FrameIdentifier, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin, const String& keySystem);

    WeakPtr<MediaKeySystemPermissionRequestManagerProxy> m_manager;
    WebCore::MediaKeySystemRequestIdentifier m_mediaKeySystemID;
    WebCore::FrameIdentifier m_mainFrameID;
    WebCore::FrameIdentifier m_frameID;
    Ref<WebCore::SecurityOrigin> m_topLevelDocumentSecurityOrigin;
    String m_keySystem;
};

} // namespace WebKit
