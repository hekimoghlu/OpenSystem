/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)

#include "SampleBufferDisplayLayer.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebKit {

class GPUProcessConnection;

class SampleBufferDisplayLayerManager : public CanMakeWeakPtr<SampleBufferDisplayLayerManager> {
    WTF_MAKE_TZONE_ALLOCATED(SampleBufferDisplayLayerManager);
public:
    SampleBufferDisplayLayerManager(GPUProcessConnection&);
    ~SampleBufferDisplayLayerManager() = default;

    void ref() const;
    void deref() const;

    void addLayer(SampleBufferDisplayLayer&);
    void removeLayer(SampleBufferDisplayLayer&);

    void didReceiveLayerMessage(IPC::Connection&, IPC::Decoder&);
    RefPtr<WebCore::SampleBufferDisplayLayer> createLayer(WebCore::SampleBufferDisplayLayerClient&);

private:
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    HashMap<SampleBufferDisplayLayerIdentifier, WeakPtr<SampleBufferDisplayLayer>> m_layers;
};

}

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)
