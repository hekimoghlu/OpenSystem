/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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

#if ENABLE(GPU_PROCESS) && HAVE(AVASSETREADER)

#include "MessageReceiver.h"
#include <WebCore/DestinationColorSpace.h>
#include <WebCore/ImageDecoderAVFObjC.h>
#include <WebCore/ImageDecoderIdentifier.h>
#include <WebCore/ProcessIdentity.h>
#include <WebCore/ShareableBitmap.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class SharedBufferReference;
}

namespace WebKit {
class GPUConnectionToWebProcess;
struct SharedPreferencesForWebProcess;

class RemoteImageDecoderAVFProxy : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteImageDecoderAVFProxy);
public:
    explicit RemoteImageDecoderAVFProxy(GPUConnectionToWebProcess&);
    virtual ~RemoteImageDecoderAVFProxy() = default;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    bool allowsExitUnderMemoryPressure() const;

    void ref() const final;
    void deref() const final;

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    void createDecoder(const IPC::SharedBufferReference&, const String& mimeType, CompletionHandler<void(std::optional<WebCore::ImageDecoderIdentifier>&&)>&&);
    void deleteDecoder(WebCore::ImageDecoderIdentifier);
    void setExpectedContentSize(WebCore::ImageDecoderIdentifier, long long expectedContentSize);
    void setData(WebCore::ImageDecoderIdentifier, const IPC::SharedBufferReference&, bool allDataReceived, CompletionHandler<void(size_t frameCount, const WebCore::IntSize& size, bool hasTrack, std::optional<Vector<WebCore::ImageDecoder::FrameInfo>>&&)>&&);
    void createFrameImageAtIndex(WebCore::ImageDecoderIdentifier, size_t index, CompletionHandler<void(std::optional<WebCore::ShareableBitmap::Handle>&&)>&&);
    void clearFrameBufferCache(WebCore::ImageDecoderIdentifier, size_t index);

    void encodedDataStatusChanged(WebCore::ImageDecoderIdentifier);

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    HashMap<WebCore::ImageDecoderIdentifier, RefPtr<WebCore::ImageDecoderAVFObjC>> m_imageDecoders;
    WebCore::ProcessIdentity m_resourceOwner;
};

}

#endif
