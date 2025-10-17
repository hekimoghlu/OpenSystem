/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "SandboxExtension.h"
#include <WebCore/MediaCanStartListener.h>
#include <WebCore/MediaKeySystemClient.h>
#include <WebCore/MediaKeySystemRequest.h>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class MediaKeySystemPermissionRequestManager : private WebCore::MediaCanStartListener {
    WTF_MAKE_TZONE_ALLOCATED(MediaKeySystemPermissionRequestManager);
public:
    explicit MediaKeySystemPermissionRequestManager(WebPage&);
    ~MediaKeySystemPermissionRequestManager() = default;

    void ref() const final;
    void deref() const final;

    void startMediaKeySystemRequest(WebCore::MediaKeySystemRequest&);
    void cancelMediaKeySystemRequest(WebCore::MediaKeySystemRequest&);
    void mediaKeySystemWasGranted(WebCore::MediaKeySystemRequestIdentifier);
    void mediaKeySystemWasDenied(WebCore::MediaKeySystemRequestIdentifier, String&&);

private:
    void sendMediaKeySystemRequest(WebCore::MediaKeySystemRequest&);

    // WebCore::MediaCanStartListener
    void mediaCanStart(WebCore::Document&) final;

    WeakRef<WebPage> m_page;

    HashMap<WebCore::MediaKeySystemRequestIdentifier, Ref<WebCore::MediaKeySystemRequest>> m_ongoingMediaKeySystemRequests;
    HashMap<RefPtr<WebCore::Document>, Vector<Ref<WebCore::MediaKeySystemRequest>>> m_pendingMediaKeySystemRequests;
};

} // namespace WebKit

#endif // ENABLE(MEDIA_STREAM)
