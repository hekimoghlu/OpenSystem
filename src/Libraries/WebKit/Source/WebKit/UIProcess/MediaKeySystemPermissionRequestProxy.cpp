/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "MediaKeySystemPermissionRequestProxy.h"

#include "MediaKeySystemPermissionRequestManagerProxy.h"
#include "WebPageProxy.h"
#include <WebCore/SecurityOrigin.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/text/StringHash.h>

namespace WebKit {
using namespace WebCore;

MediaKeySystemPermissionRequestProxy::MediaKeySystemPermissionRequestProxy(MediaKeySystemPermissionRequestManagerProxy& manager, MediaKeySystemRequestIdentifier mediaKeySystemID, FrameIdentifier mainFrameID, FrameIdentifier frameID, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin, const String& keySystem)
    : m_manager(manager)
    , m_mediaKeySystemID(mediaKeySystemID)
    , m_mainFrameID(mainFrameID)
    , m_frameID(frameID)
    , m_topLevelDocumentSecurityOrigin(WTFMove(topLevelDocumentOrigin))
    , m_keySystem(keySystem)
{
}

void MediaKeySystemPermissionRequestProxy::allow()
{
    RefPtr manager = m_manager.get();
    ASSERT(manager);
    if (!manager)
        return;

    manager->grantRequest(*this);
    invalidate();
}

void MediaKeySystemPermissionRequestProxy::deny()
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return;

    manager->denyRequest(*this);
    invalidate();
}

void MediaKeySystemPermissionRequestProxy::invalidate()
{
    m_manager = nullptr;
}

void MediaKeySystemPermissionRequestProxy::doDefaultAction()
{
#if PLATFORM(COCOA)
    // Backward compatibility for platforms not yet supporting the MediaKeySystem permission request
    // in their APIUIClient.
    allow();
#else
    deny();
#endif
}

} // namespace WebKit
