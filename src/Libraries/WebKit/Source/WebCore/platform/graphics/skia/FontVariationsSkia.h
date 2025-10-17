/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#include "FontDescription.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN // GLib / Win port
#include <skia/core/SkTypeface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

struct FontVariationDefaults {
    float clamp(float value) const
    {
        ASSERT(minimumValue <= maximumValue);
        return std::clamp(value, minimumValue, maximumValue);
    }

    String axisName;
    float defaultValue;
    float minimumValue;
    float maximumValue;
};

typedef UncheckedKeyHashMap<FontTag, FontVariationDefaults, FourCharacterTagHash, FourCharacterTagHashTraits> FontVariationDefaultsMap;
FontVariationDefaultsMap defaultFontVariationValues(const SkTypeface&);

} // namespace WebCore
