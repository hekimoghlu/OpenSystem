/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#include <pal/spi/cf/CoreTextSPI.h>

using CharacterClass = uint8_t;

namespace CompositionRules {

typedef enum : uint8_t {
    _______ = 0,
    _1_4_be,
    _1_4_af,
    _1_8_be,
    _1_8_af,
    _1_25_be,
    _1_2_be,
    _1_2_af,
    _note3_,
    _note5_,
    _1_4_af_re, // Reduce 1/4 em after previous char.
    _1_4_be_re, // Reduce 1/4 em before current char.
    _1_2_eq_re, // Reduce 1/4 em after previous char and reduce 1/4 em before current char.
} CharacterSpacingType;

CharacterSpacingType characterSpacing(CTCompositionLanguage, bool, UTF32Char, UTF32Char);

} // namespace CompositionRules
