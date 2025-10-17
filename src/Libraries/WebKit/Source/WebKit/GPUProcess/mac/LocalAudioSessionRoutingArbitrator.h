/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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

#if ENABLE(ROUTING_ARBITRATION) && HAVE(AVAUDIO_ROUTING_ARBITER)

#include "AudioSessionRoutingArbitratorProxy.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakRef.h>

namespace WTF {
class Logger;
}

namespace WebKit {

class GPUConnectionToWebProcess;

class LocalAudioSessionRoutingArbitrator final
    : public WebCore::AudioSessionRoutingArbitrationClient {
    WTF_MAKE_TZONE_ALLOCATED(LocalAudioSessionRoutingArbitrator);

    friend UniqueRef<LocalAudioSessionRoutingArbitrator> WTF::makeUniqueRefWithoutFastMallocCheck<LocalAudioSessionRoutingArbitrator>(GPUConnectionToWebProcess&);
public:
    USING_CAN_MAKE_WEAKPTR(WebCore::AudioSessionRoutingArbitrationClient);

    static std::unique_ptr<LocalAudioSessionRoutingArbitrator> create(GPUConnectionToWebProcess&);
    LocalAudioSessionRoutingArbitrator(GPUConnectionToWebProcess&);
    virtual ~LocalAudioSessionRoutingArbitrator();

    void processDidTerminate();

private:

    // AudioSessionRoutingArbitrationClient
    void beginRoutingArbitrationWithCategory(WebCore::AudioSession::CategoryType, ArbitrationCallback&&) final;
    void leaveRoutingAbritration() final;

    Logger& logger();
    ASCIILiteral logClassName() const { return "LocalAudioSessionRoutingArbitrator"_s; }
    WTFLogChannel& logChannel() const;
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    bool canLog() const final;

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    const uint64_t m_logIdentifier;
};

}

#endif
