/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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

#include <wtf/ObjectIdentifier.h>

namespace WebKit {

struct WebExtensionTabIdentifierType;
using WebExtensionTabIdentifier = ObjectIdentifier<WebExtensionTabIdentifierType>;

namespace WebExtensionTabConstants {

    static constexpr double None { -1 };

    static constexpr const WebExtensionTabIdentifier NoneIdentifier { std::numeric_limits<uint64_t>::max() - 1 };

}

inline bool isNone(WebExtensionTabIdentifier identifier)
{
    return identifier == WebExtensionTabConstants::NoneIdentifier;
}

inline bool isNone(std::optional<WebExtensionTabIdentifier> identifier)
{
    return identifier && isNone(identifier.value());
}

inline bool isValid(std::optional<WebExtensionTabIdentifier> identifier)
{
    return identifier && !isNone(identifier.value());
}

inline std::optional<WebExtensionTabIdentifier> toWebExtensionTabIdentifier(double identifier)
{
    if (identifier == WebExtensionTabConstants::None)
        return WebExtensionTabConstants::NoneIdentifier;

    if (!std::isfinite(identifier) || identifier <= 0 || identifier >= static_cast<double>(WebExtensionTabConstants::NoneIdentifier.toUInt64()))
        return std::nullopt;

    double integral;
    if (std::modf(identifier, &integral) != 0.0) {
        // Only integral numbers can be used.
        return std::nullopt;
    }

    auto identifierAsUint64 = static_cast<uint64_t>(identifier);
    if (!WebExtensionTabIdentifier::isValidIdentifier(identifierAsUint64)) {
        ASSERT_NOT_REACHED();
        return WebExtensionTabConstants::NoneIdentifier;
    }

    return WebExtensionTabIdentifier { identifierAsUint64 };
}

inline double toWebAPI(const WebExtensionTabIdentifier& identifier)
{
    if (isNone(identifier))
        return WebExtensionTabConstants::None;

    return static_cast<double>(identifier.toUInt64());
}

}
