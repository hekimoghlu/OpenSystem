/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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

#if ENABLE(GPU_PROCESS)

#include "RemoteDeviceProxy.h"
#include "RemoteGPUProxy.h"
#include "RemotePresentationContextProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUXREye.h>
#include <WebCore/WebGPUXRView.h>

namespace WebCore {
class WebXRFrame;
}

namespace WebCore::WebGPU {
class Device;
class XRProjectionLayer;
class XRView;
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteXRViewProxy final : public WebCore::WebGPU::XRView {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRViewProxy);
public:
    static Ref<RemoteXRViewProxy> create(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRViewProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteXRViewProxy();

    RemoteDeviceProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }

private:
    friend class DowncastConvertToBackingContext;

    RemoteXRViewProxy(RemoteDeviceProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteXRViewProxy(const RemoteXRViewProxy&) = delete;
    RemoteXRViewProxy(RemoteXRViewProxy&&) = delete;
    RemoteXRViewProxy& operator=(const RemoteXRViewProxy&) = delete;
    RemoteXRViewProxy& operator=(RemoteXRViewProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }

    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }
    template<typename T>
    WARN_UNUSED_RETURN IPC::Connection::SendSyncResult<T> sendSync(T&& message)
    {
        return root().protectedStreamClientConnection()->sendSync(WTFMove(message), backing());
    }

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteDeviceProxy> m_parent;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
