/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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

#include "FontTaggedSettings.h"
#include "ShouldLocalizeAxisNames.h"
#include <wtf/HashMap.h>

typedef struct FT_FaceRec_* FT_Face;

namespace WebCore {
class FontDescription;

#if ENABLE(VARIATION_FONTS)
struct VariationDefaults {
    String axisName;
    float defaultValue;
    float minimumValue;
    float maximumValue;
};

typedef UncheckedKeyHashMap<FontTag, VariationDefaults, FourCharacterTagHash, FourCharacterTagHashTraits> VariationDefaultsMap;
typedef UncheckedKeyHashMap<FontTag, float, FourCharacterTagHash, FourCharacterTagHashTraits> VariationsMap;

VariationDefaultsMap defaultVariationValues(FT_Face, ShouldLocalizeAxisNames);

String buildVariationSettings(FT_Face, const FontDescription&, const FontCreationContext&);
#endif
};
