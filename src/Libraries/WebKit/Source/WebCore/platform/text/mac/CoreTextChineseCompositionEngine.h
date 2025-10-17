/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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

#include "CoreTextCompositionEngine.h"
#include <pal/spi/cf/CoreTextSPI.h>

class ChineseCompositionRules {
public:
    typedef enum : uint8_t {
        Opening = 0,
        Closing,
        Whitespace,
        FullWidth,
        HalfWidth,
        HalfWidthOpening,
        HalfWidthClosing,
        Centered,
        Other,

        NumClasses
    } ChineseCharacterClass;

    static ChineseCharacterClass characterClass(UTF32Char, uint32_t, CTCompositionLanguage);
    static CompositionRules::CharacterSpacingType characterSpacing(CTCompositionLanguage, bool, UTF32Char, UTF32Char);
};
