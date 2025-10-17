/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

#if ENABLE(ROUTING_ARBITRATION)

#include <WebCore/AudioSession.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class WebProcess;

class AudioSessionRoutingArbitrator final : public WebCore::AudioSessionRoutingArbitrationClient {
    WTF_MAKE_TZONE_ALLOCATED(AudioSessionRoutingArbitrator);
public:
    USING_CAN_MAKE_WEAKPTR(WebCore::AudioSessionRoutingArbitrationClient);

    explicit AudioSessionRoutingArbitrator(WebProcess&);
    virtual ~AudioSessionRoutingArbitrator();

    static ASCIILiteral supplementName();

    // AudioSessionRoutingAbritrator
    void beginRoutingArbitrationWithCategory(WebCore::AudioSession::CategoryType, CompletionHandler<void(RoutingArbitrationError, DefaultRouteChanged)>&&) final;
    void leaveRoutingAbritration() final;

private:
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    bool canLog() const final;

    WebCore::AudioSession::ChangedObserver m_observer;
    const uint64_t m_logIdentifier;
};

}

#endif
