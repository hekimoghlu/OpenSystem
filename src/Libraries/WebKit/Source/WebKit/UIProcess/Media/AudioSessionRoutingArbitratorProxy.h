/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#include "MessageReceiver.h"
#include <WebCore/AudioSession.h>
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/WeakPtr.h>

#if HAVE(AVAUDIO_ROUTING_ARBITER)
#import <WebCore/SharedRoutingArbitrator.h>
#endif

namespace WebKit {

class WebProcessProxy;
struct SharedPreferencesForWebProcess;

class AudioSessionRoutingArbitratorProxy
    : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(AudioSessionRoutingArbitratorProxy);
public:
    explicit AudioSessionRoutingArbitratorProxy(WebProcessProxy&);
    virtual ~AudioSessionRoutingArbitratorProxy();

    void processDidTerminate();
    WebCore::AudioSession::CategoryType category() const { return m_category; }

    static uint64_t destinationId() { return 1; }

    using RoutingArbitrationError = WebCore::AudioSessionRoutingArbitrationClient::RoutingArbitrationError;
    using DefaultRouteChanged = WebCore::AudioSessionRoutingArbitrationClient::DefaultRouteChanged;
    using ArbitrationCallback = WebCore::AudioSessionRoutingArbitrationClient::ArbitrationCallback;

    enum class ArbitrationStatus : uint8_t {
        None,
        Pending,
        Active,
    };

    ArbitrationStatus arbitrationStatus() const { return m_arbitrationStatus; }
    WallTime arbitrationUpdateTime() const { return m_arbitrationUpdateTime; }
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    void ref() const final;
    void deref() const final;

protected:
    Logger& logger();
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "AudioSessionRoutingArbitrator"_s; }
    WTFLogChannel& logChannel() const;

private:
    Ref<WebProcessProxy> protectedProcess();

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void beginRoutingArbitrationWithCategory(WebCore::AudioSession::CategoryType, ArbitrationCallback&&);
    void endRoutingArbitration();

    WeakRef<WebProcessProxy> m_process;
    WebCore::AudioSession::CategoryType m_category { WebCore::AudioSession::CategoryType::None };
    ArbitrationStatus m_arbitrationStatus { ArbitrationStatus::None };
    WallTime m_arbitrationUpdateTime;
    uint64_t m_logIdentifier { 0 };

#if HAVE(AVAUDIO_ROUTING_ARBITER)
    UniqueRef<WebCore::SharedRoutingArbitratorToken> m_token;
#endif
};

}

#endif
