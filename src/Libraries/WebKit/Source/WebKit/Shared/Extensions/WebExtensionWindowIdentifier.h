/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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

#include <wtf/Assertions.h>
#include <wtf/ObjectIdentifier.h>

namespace WebKit {

struct WebExtensionWindowIdentifierType;
using WebExtensionWindowIdentifier = ObjectIdentifier<WebExtensionWindowIdentifierType>;

namespace WebExtensionWindowConstants {

    static constexpr double None { -1 };
    static constexpr double Current { -2 };

    static constexpr const WebExtensionWindowIdentifier NoneIdentifier { std::numeric_limits<uint64_t>::max() - 1 };
    static constexpr const WebExtensionWindowIdentifier CurrentIdentifier { std::numeric_limits<uint64_t>::max() - 2 };

}

inline bool isNone(WebExtensionWindowIdentifier identifier)
{
    return identifier == WebExtensionWindowConstants::NoneIdentifier;
}

inline bool isNone(std::optional<WebExtensionWindowIdentifier> identifier)
{
    return identifier && isNone(identifier.value());
}

inline bool isCurrent(WebExtensionWindowIdentifier identifier)
{
    return identifier == WebExtensionWindowConstants::CurrentIdentifier;
}

inline bool isCurrent(std::optional<WebExtensionWindowIdentifier> identifier)
{
    return identifier && isCurrent(identifier.value());
}

inline bool isValid(std::optional<WebExtensionWindowIdentifier> identifier)
{
    return identifier && !isNone(identifier.value());
}

inline std::optional<WebExtensionWindowIdentifier> toWebExtensionWindowIdentifier(double identifier)
{
    if (identifier == WebExtensionWindowConstants::None)
        return WebExtensionWindowConstants::NoneIdentifier;

    if (identifier == WebExtensionWindowConstants::Current)
        return WebExtensionWindowConstants::CurrentIdentifier;

    if (!std::isfinite(identifier) || identifier <= 0 || identifier >= static_cast<double>(WebExtensionWindowConstants::CurrentIdentifier.toUInt64()))
        return std::nullopt;

    double integral;
    if (std::modf(identifier, &integral) != 0.0) {
        // Only integral numbers can be used.
        return std::nullopt;
    }

    auto identifierAsUInt64 = static_cast<uint64_t>(identifier);
    if (!WebExtensionWindowIdentifier::isValidIdentifier(identifierAsUInt64)) {
        ASSERT_NOT_REACHED();
        return WebExtensionWindowConstants::NoneIdentifier;
    }

    return WebExtensionWindowIdentifier { identifierAsUInt64 };
}

inline double toWebAPI(const WebExtensionWindowIdentifier& identifier)
{
    if (isNone(identifier))
        return WebExtensionWindowConstants::None;

    if (isCurrent(identifier)) {
        ASSERT_NOT_REACHED_WITH_MESSAGE("The current window identifier should not be returned to JavaScript. It is only an input value.");
        return WebExtensionWindowConstants::None;
    }

    return static_cast<double>(identifier.toUInt64());
}

}
