/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

#include "DownloadID.h"
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(IOS_FAMILY)
#include <objc/objc.h>
#endif

namespace API {
class DownloadClient;
}

namespace WebCore {
class ResourceRequest;
}

namespace WebKit {

class DownloadProxy;
class NetworkProcessProxy;
class ProcessAssertion;
class WebPageProxy;
class WebsiteDataStore;
struct FrameInfoData;

class DownloadProxyMap : public CanMakeWeakPtr<DownloadProxyMap> {
    WTF_MAKE_TZONE_ALLOCATED(DownloadProxyMap);
    WTF_MAKE_NONCOPYABLE(DownloadProxyMap);
public:
    explicit DownloadProxyMap(NetworkProcessProxy&);
    ~DownloadProxyMap();

    Ref<DownloadProxy> createDownloadProxy(WebsiteDataStore&, Ref<API::DownloadClient>&&, const WebCore::ResourceRequest&, const FrameInfoData&, WebPageProxy* originatingPage);
    void downloadFinished(DownloadProxy&);

    bool isEmpty() const { return m_downloads.isEmpty(); }
    void invalidate();

    void ref() const;
    void deref() const;

private:
    Ref<NetworkProcessProxy> protectedProcess();

    void platformCreate();
    void platformDestroy();

    WeakRef<NetworkProcessProxy> m_process;
    HashMap<DownloadID, RefPtr<DownloadProxy>> m_downloads;

    bool m_shouldTakeAssertion { false };
    RefPtr<ProcessAssertion> m_downloadUIAssertion;
};

} // namespace WebKit
