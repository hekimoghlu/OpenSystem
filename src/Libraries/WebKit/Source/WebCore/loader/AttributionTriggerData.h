/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

#include "EphemeralNonce.h"
#include "PCMTokens.h"
#include "RegistrableDomain.h"
#include <wtf/JSONValues.h>
#include <wtf/URL.h>

namespace WebCore::PCM {

enum class WasSent : bool { No, Yes };

struct AttributionTriggerData {
    static constexpr uint8_t MaxEntropy = 15;

    struct Priority {
        static constexpr uint8_t MaxEntropy = 63;
        using PriorityValue = uint8_t;

        explicit Priority(PriorityValue value)
            : value { value }
        {
        }
        
        PriorityValue value { 0 };
    };

    bool isValid() const
    {
        return data <= MaxEntropy && priority <= Priority::MaxEntropy;
    }

    void setDestinationUnlinkableTokenValue(const String& value)
    {
        if (!destinationUnlinkableToken)
            destinationUnlinkableToken = DestinationUnlinkableToken { };
        destinationUnlinkableToken->valueBase64URL = value;
    }
    void setDestinationSecretToken(const DestinationSecretToken& token) { destinationSecretToken = token; }
    WEBCORE_EXPORT const std::optional<const URL> tokenPublicKeyURL() const;
    WEBCORE_EXPORT const std::optional<const URL> tokenSignatureURL() const;
    WEBCORE_EXPORT Ref<JSON::Object> tokenSignatureJSON() const;

    uint8_t data { 0 };
    Priority::PriorityValue priority;
    WasSent wasSent { WasSent::No };
    std::optional<RegistrableDomain> sourceRegistrableDomain;
    std::optional<EphemeralNonce> ephemeralDestinationNonce;
    std::optional<RegistrableDomain> destinationSite;

    // These values are not serialized.
    std::optional<DestinationUnlinkableToken> destinationUnlinkableToken { };
    std::optional<DestinationSecretToken> destinationSecretToken { };
};

}
