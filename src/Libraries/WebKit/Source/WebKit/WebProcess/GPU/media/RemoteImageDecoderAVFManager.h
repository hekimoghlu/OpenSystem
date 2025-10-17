/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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

#include "Connection.h"
#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include <WebCore/ImageDecoderIdentifier.h>
#include <WebCore/ImageTypes.h>
#include <WebCore/IntSize.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class RemoteImageDecoderAVF;
class WebProcess;

class RemoteImageDecoderAVFManager final
    : private GPUProcessConnection::Client
    , private IPC::MessageReceiver
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteImageDecoderAVFManager> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteImageDecoderAVFManager);
public:
    static Ref<RemoteImageDecoderAVFManager> create();
    virtual ~RemoteImageDecoderAVFManager();

    void deleteRemoteImageDecoder(const WebCore::ImageDecoderIdentifier&);

    void setUseGPUProcess(bool);
    GPUProcessConnection& ensureGPUProcessConnection();

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    RemoteImageDecoderAVFManager();
    RefPtr<RemoteImageDecoderAVF> createImageDecoder(WebCore::FragmentedSharedBuffer& data, const String& mimeType, WebCore::AlphaOption, WebCore::GammaAndColorProfileOption);

    // GPUProcessConnection::Client.
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    void encodedDataStatusChanged(const WebCore::ImageDecoderIdentifier&, size_t frameCount, const WebCore::IntSize&, bool hasTrack);

    HashMap<WebCore::ImageDecoderIdentifier, WeakPtr<RemoteImageDecoderAVF>> m_remoteImageDecoders;

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
};

}

#endif
