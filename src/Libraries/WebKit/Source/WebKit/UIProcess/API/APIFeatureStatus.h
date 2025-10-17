/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#include <optional>
#include <type_traits>

namespace API {
enum class FeatureStatus : uint8_t {
    // For customizing WebKit behavior in embedding applications.
    Embedder,
    // Feature in active development. Unfinished, no promise it is usable or safe.
    Unstable,
    // Tools for debugging the WebKit engine. Not generally useful to web developers.
    Internal,
    // Tools for web developers.
    Developer,
    // Enabled by default in test infrastructure, but not ready to ship yet.
    Testable,
    // Enabled by default in Safari Technology Preview, but not considered ready to ship yet.
    Preview,
    // Enabled by default and ready for general use.
    Stable,
    // Enabled by default and in general use for more than a year.
    Mature
};

// Helper for representing feature status as a constant type. Used by the preferences generator to
// validate feature configuration at compile time.
template<API::FeatureStatus Status>
class FeatureConstant : public std::integral_constant<API::FeatureStatus, Status> { };

constexpr std::optional<bool> defaultValueForFeatureStatus(FeatureStatus status)
{
    switch (status) {
    case FeatureStatus::Stable:
    case FeatureStatus::Mature:
        return true;
    case FeatureStatus::Unstable:
    case FeatureStatus::Developer:
    case FeatureStatus::Testable:
    case FeatureStatus::Preview:
        return false;
    case FeatureStatus::Embedder:
    case FeatureStatus::Internal:
        // Embedder features vary widely between platforms, so they have no
        // implied default.
        // FIXME: Internal features should be off by default, but they need
        // additional auditing.
        return { };
    }
}

enum class FeatureCategory : uint8_t {
    None,
    Animation,
    CSS,
    DOM,
    Extensions,
    HTML,
    Javascript,
    Media,
    Networking,
    Privacy,
    Security
};

}
