/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

#if USE(GRAPHICS_LAYER_WC)

#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include "MessageSender.h"
#include "UpdateInfo.h"
#include "WCLayerTreeHostIdentifier.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

struct WCUpdateInfo;

class RemoteWCLayerTreeHostProxy
    : private IPC::MessageSender
    , private GPUProcessConnection::Client
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteWCLayerTreeHostProxy> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteWCLayerTreeHostProxy);
public:
    RemoteWCLayerTreeHostProxy(WebPage&, bool usesOffscreenRendering);
    ~RemoteWCLayerTreeHostProxy();

    void update(WCUpdateInfo&&, CompletionHandler<void(std::optional<WebKit::UpdateInfo>)>&&);

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    WCLayerTreeHostIdentifier wcLayerTreeHostIdentifier() const { return m_wcLayerTreeHostIdentifier; };
    GPUProcessConnection& ensureGPUProcessConnection();
    void disconnectGpuProcessIfNeeded();

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    // IPC::MessageSender
    IPC::Connection* messageSenderConnection() const override;
    uint64_t messageSenderDestinationID() const override;

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    WCLayerTreeHostIdentifier m_wcLayerTreeHostIdentifier { WCLayerTreeHostIdentifier::generate() };
    WeakRef<WebPage> m_page;
    bool m_usesOffscreenRendering { false };
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
