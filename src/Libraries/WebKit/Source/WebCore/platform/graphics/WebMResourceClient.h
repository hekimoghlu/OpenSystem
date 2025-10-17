/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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

#if ENABLE(ALTERNATE_WEBM_PLAYER)

#include "PlatformMediaResourceLoader.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class WebMResourceClientParent : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<WebMResourceClientParent, WTF::DestructionThread::Main> {
public:
    virtual ~WebMResourceClientParent() = default;

    virtual void dataReceived(const SharedBuffer&) = 0;
    virtual void loadFailed(const ResourceError&) = 0;
    virtual void loadFinished() = 0;
    virtual void dataLengthReceived(size_t) = 0;
};

class WebMResourceClient final
    : public PlatformMediaResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(WebMResourceClient);
public:
    static RefPtr<WebMResourceClient> create(WebMResourceClientParent&, PlatformMediaResourceLoader&, ResourceRequest&&);
    ~WebMResourceClient() { stop(); }

    void stop();

private:
    WebMResourceClient(WebMResourceClientParent&, Ref<PlatformMediaResource>&&);

    void responseReceived(PlatformMediaResource&, const ResourceResponse&, CompletionHandler<void(ShouldContinuePolicyCheck)>&&) final;
    void dataReceived(PlatformMediaResource&, const SharedBuffer&) final;
    void loadFailed(PlatformMediaResource&, const ResourceError&) final;
    void loadFinished(PlatformMediaResource&, const NetworkLoadMetrics&) final;

    ThreadSafeWeakPtr<WebMResourceClientParent> m_parent;
    RefPtr<PlatformMediaResource> m_resource;
};

} // namespace WebCore

#endif // ENABLE(ALTERNATE_WEBM_PLAYER)
