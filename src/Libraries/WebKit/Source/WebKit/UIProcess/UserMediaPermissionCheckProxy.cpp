/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#include "UserMediaPermissionCheckProxy.h"

#include "APIUIClient.h"
#include "UserMediaPermissionRequestManagerProxy.h"
#include <WebCore/SecurityOrigin.h>
#include <WebCore/SecurityOriginData.h>

namespace WebKit {
using namespace WebCore;

UserMediaPermissionCheckProxy::UserMediaPermissionCheckProxy(FrameIdentifier frameID, CompletionHandler&& handler, Ref<WebCore::SecurityOrigin>&& userMediaDocumentOrigin, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin)
    : m_frameID(frameID)
    , m_completionHandler(WTFMove(handler))
    , m_userMediaDocumentSecurityOrigin(WTFMove(userMediaDocumentOrigin))
    , m_topLevelDocumentSecurityOrigin(WTFMove(topLevelDocumentOrigin))
{
}

UserMediaPermissionCheckProxy::~UserMediaPermissionCheckProxy()
{
    invalidate();
}

void UserMediaPermissionCheckProxy::setUserMediaAccessInfo(bool allowed)
{
    ASSERT(m_completionHandler);
    complete(allowed ? PermissionInfo::Granted : PermissionInfo::Unknown);
}

void UserMediaPermissionCheckProxy::complete(PermissionInfo info)
{
    if (!m_completionHandler)
        return;

    auto completionHandler = WTFMove(m_completionHandler);
    completionHandler(info);
}

} // namespace WebKit
