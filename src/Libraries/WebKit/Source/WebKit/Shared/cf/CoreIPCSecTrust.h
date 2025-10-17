/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#if USE(CF)

#import "CoreIPCData.h"
#import "CoreIPCDate.h"
#import "CoreIPCNumber.h"
#import "CoreIPCString.h"

#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/cf/VectorCF.h>
#import <wtf/spi/cocoa/SecuritySPI.h>

namespace WebKit {

#if HAVE(WK_SECURE_CODING_SECTRUST)

enum class CoreIPCSecTrustResult : uint8_t {
    Invalid = 0,
    Proceed,
    Confirm,
    Deny,
    Unspecified,
    RecoverableTrustFailure,
    FatalTrustFailure,
    OtherError
};

struct CoreIPCSecTrustData {
    using Detail = Vector<std::pair<CoreIPCString, bool>>;
    using InfoOption = std::variant<CoreIPCDate, CoreIPCString, bool>;
    using InfoType = Vector<std::pair<CoreIPCString, InfoOption>>;
    using PolicyDictionaryValueIsNumber = Vector<std::pair<CoreIPCString, CoreIPCNumber>>;
    using PolicyArrayOfArrayContainingDateOrNumbers = Vector<Vector<std::variant<CoreIPCNumber, CoreIPCDate>>>;
    using PolicyArrayOfNumbers = Vector<CoreIPCNumber>;
    using PolicyArrayOfStrings = Vector<CoreIPCString>;
    using PolicyArrayOfData = Vector<CoreIPCData>;
    using PolicyVariant = std::variant<bool, CoreIPCString, PolicyArrayOfNumbers, PolicyArrayOfStrings, PolicyArrayOfData, PolicyArrayOfArrayContainingDateOrNumbers, PolicyDictionaryValueIsNumber>;
    using PolicyOption = Vector<std::pair<CoreIPCString, PolicyVariant>>;
    using PolicyValue = std::variant<CoreIPCString, PolicyOption>;
    using PolicyType = Vector<std::pair<CoreIPCString, PolicyValue>>;
    using ExceptionType = Vector<std::pair<CoreIPCString, std::variant<CoreIPCNumber, CoreIPCData, bool>>>;

    CoreIPCSecTrustResult result { CoreIPCSecTrustResult::Invalid };
    bool anchorsOnly { false };
    bool keychainsAllowed { false };
    Vector<CoreIPCData> certificates;
    Vector<CoreIPCData> chain;
    Vector<Detail> details;
    Vector<PolicyType> policies;
    std::optional<InfoType> info;
    std::optional<CoreIPCDate> verifyDate;
    std::optional<Vector<CoreIPCData>> responses;
    std::optional<Vector<CoreIPCData>> scts;
    std::optional<Vector<CoreIPCData>> anchors;
    std::optional<Vector<CoreIPCData>> trustedLogs;
    std::optional<Vector<ExceptionType>> exceptions;
};

class CoreIPCSecTrust {
    WTF_MAKE_TZONE_ALLOCATED(CoreIPCSecTrust);
public:
    CoreIPCSecTrust() { }

    CoreIPCSecTrust(SecTrustRef);

    CoreIPCSecTrust(std::optional<WebKit::CoreIPCSecTrustData>&& data)
        : m_data(WTFMove(data)) { }

    RetainPtr<SecTrustRef> createSecTrust() const;

    std::optional<CoreIPCSecTrustData> m_data;

    enum class PolicyOptionValueShape {
        Invalid,
        Bool,
        String,
        ArrayOfNumbers,
        ArrayOfStrings,
        ArrayOfData,
        ArrayOfArrayContainingDateOrNumber,
        DictionaryValueIsNumber,
    };
    static PolicyOptionValueShape detectPolicyOptionShape(id);
};
#else // !HAVE(WK_SECURE_CODING_SECTRUST)
class CoreIPCSecTrust {
public:
    CoreIPCSecTrust()
        : m_trustData() { };

    CoreIPCSecTrust(SecTrustRef trust)
        : m_trustData(adoptCF(SecTrustSerialize(trust, NULL)))
    {
    }

    CoreIPCSecTrust(RetainPtr<CFDataRef> data)
        : m_trustData(data)
    {
    }

    CoreIPCSecTrust(std::span<const uint8_t> data)
        : m_trustData(data.empty() ? nullptr : adoptCF(CFDataCreate(kCFAllocatorDefault, data.data(), data.size())))
    {
    }

    RetainPtr<SecTrustRef> createSecTrust() const
    {
        if (!m_trustData)
            return nullptr;

        return adoptCF(SecTrustDeserialize(m_trustData.get(), NULL));
    }

    std::span<const uint8_t> dataReference() const
    {
        if (!m_trustData)
            return { };

        return span(m_trustData.get());
    }

private:
    RetainPtr<CFDataRef> m_trustData;
};
#endif

} // namespace WebKit

#endif // USE(CF)
