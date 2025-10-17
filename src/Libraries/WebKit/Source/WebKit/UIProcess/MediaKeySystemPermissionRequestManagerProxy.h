/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

#include "MediaKeySystemPermissionRequestProxy.h"
#include <WebCore/SecurityOrigin.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/LoggerHelper.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SecurityOrigin;
};

namespace WebKit {

class WebPageProxy;

class MediaKeySystemPermissionRequestManagerProxy : public CanMakeWeakPtr<MediaKeySystemPermissionRequestManagerProxy> {
    WTF_MAKE_TZONE_ALLOCATED(MediaKeySystemPermissionRequestManagerProxy);
public:
    explicit MediaKeySystemPermissionRequestManagerProxy(WebPageProxy&);
    ~MediaKeySystemPermissionRequestManagerProxy();

    WebPageProxy& page() const { return m_page.get(); }

    void invalidatePendingRequests();

    Ref<MediaKeySystemPermissionRequestProxy> createRequestForFrame(WebCore::MediaKeySystemRequestIdentifier, WebCore::FrameIdentifier, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin, const String& keySystem);

    void grantRequest(MediaKeySystemPermissionRequestProxy&);
    void denyRequest(MediaKeySystemPermissionRequestProxy&, const String& message = { });

    void ref() const;
    void deref() const;

private:
#if !RELEASE_LOG_DISABLED
    const Logger& logger() const;
    uint64_t logIdentifier() const { return m_logIdentifier; }
#endif

    WeakRef<WebPageProxy> m_page;

    HashMap<WebCore::MediaKeySystemRequestIdentifier, RefPtr<MediaKeySystemPermissionRequestProxy>> m_pendingRequests;
    HashSet<String> m_validAuthorizationTokens;

#if !RELEASE_LOG_DISABLED
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebKit
