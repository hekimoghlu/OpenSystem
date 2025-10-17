/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#include "AudioSession.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
class SharedRoutingArbitratorToken;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SharedRoutingArbitratorToken> : std::true_type { };
}

namespace WTF {
class Logger;
}

namespace WebCore {

class WEBCORE_EXPORT SharedRoutingArbitratorToken : public CanMakeWeakPtr<SharedRoutingArbitratorToken> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SharedRoutingArbitratorToken, WEBCORE_EXPORT);
public:
    static UniqueRef<SharedRoutingArbitratorToken> create();
    uint64_t logIdentifier() const;
private:
    friend UniqueRef<SharedRoutingArbitratorToken> WTF::makeUniqueRefWithoutFastMallocCheck<SharedRoutingArbitratorToken>();
    SharedRoutingArbitratorToken() = default;
    mutable uint64_t m_logIdentifier { 0 };
};

class WEBCORE_EXPORT SharedRoutingArbitrator {
public:
    static SharedRoutingArbitrator& sharedInstance();

    using RoutingArbitrationError = AudioSessionRoutingArbitrationClient::RoutingArbitrationError;
    using DefaultRouteChanged = AudioSessionRoutingArbitrationClient::DefaultRouteChanged;
    using ArbitrationCallback = AudioSessionRoutingArbitrationClient::ArbitrationCallback;

    bool isInRoutingArbitrationForToken(const SharedRoutingArbitratorToken&);
    void beginRoutingArbitrationForToken(const SharedRoutingArbitratorToken&, AudioSession::CategoryType, ArbitrationCallback&&);
    void endRoutingArbitrationForToken(const SharedRoutingArbitratorToken&);

    void setLogger(const Logger&);

private:
    const Logger& logger();
    ASCIILiteral logClassName() const { return "SharedRoutingArbitrator"_s; }
    WTFLogChannel& logChannel() const;

    std::optional<AudioSession::CategoryType> m_currentCategory { AudioSession::CategoryType::None };
    WeakHashSet<SharedRoutingArbitratorToken> m_tokens;
    Vector<ArbitrationCallback> m_enqueuedCallbacks;
    RefPtr<const Logger> m_logger;
    bool m_setupArbitrationOngoing { false };
};

}

#endif
