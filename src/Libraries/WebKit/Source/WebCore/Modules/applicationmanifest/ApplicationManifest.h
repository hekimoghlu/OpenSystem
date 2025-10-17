/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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

#if ENABLE(APPLICATION_MANIFEST)

#include "Color.h"
#include "ScreenOrientationLockType.h"
#include <optional>
#include <wtf/URL.h>
#include <wtf/Vector.h>

namespace WebCore {

struct ApplicationManifest {
    enum class Direction : uint8_t {
        Auto,
        LTR, // NOLINT
        RTL, // NOLINT
    };

    enum class Display : uint8_t {
        Browser,
        MinimalUI,
        Standalone,
        Fullscreen,
    };

    struct Icon {
        enum class Purpose : uint8_t {
            Any = 1 << 0,
            Monochrome = 1 << 1,
            Maskable = 1 << 2,
        };

        URL src;
        Vector<String> sizes;
        String type;
        OptionSet<Purpose> purposes;
    };

    struct Shortcut {
        String name;
        URL url;
        Vector<Icon> icons;
    };

    String rawJSON;
    Direction dir;
    String name;
    String shortName;
    String description;
    URL scope;
    bool isDefaultScope { false };
    Display display;
    std::optional<ScreenOrientationLockType> orientation;
    URL manifestURL;
    URL startURL;
    URL id;
    Color backgroundColor;
    Color themeColor;
    Vector<String> categories;
    Vector<Icon> icons;
    Vector<Shortcut> shortcuts;
};

} // namespace WebCore

#endif // ENABLE(APPLICATION_MANIFEST)

