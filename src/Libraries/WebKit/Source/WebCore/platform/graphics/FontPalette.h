/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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

#include <variant>
#include <wtf/Hasher.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

struct FontPalette {
    bool operator==(const FontPalette& other) const
    {
        if (type == Type::Custom)
            return other.type == Type::Custom && identifier == other.identifier;
        return type == other.type;
    }

    enum class Type : uint8_t {
        Normal,
        Light,
        Dark,
        Custom
    } type;

    AtomString identifier;
};

inline void add(Hasher& hasher, const FontPalette& request)
{
    add(hasher, request.type);
    if (request.type == FontPalette::Type::Custom)
        add(hasher, request.identifier);
}

inline TextStream& operator<<(TextStream& ts, const FontPalette& fontPalette)
{
    switch (fontPalette.type) {
    case FontPalette::Type::Normal:
        ts << "normal";
        break;
    case FontPalette::Type::Light:
        ts << "light";
        break;
    case FontPalette::Type::Dark:
        ts << "dark";
        break;
    case FontPalette::Type::Custom:
        ts << "custom: " << fontPalette.identifier;
        break;
    }
    return ts;
}

}
