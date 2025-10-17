/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

#include <array>

namespace WTF {

// Takes a six-character string that encodes a 32-bit integer, and returns that
// integer. RELEASE_ASSERT's that the string represents a valid six-character
// hash.
WTF_EXPORT_PRIVATE unsigned sixCharacterHashStringToInteger(std::span<const char, 6>);

// Takes a 32-bit integer and constructs a six-character string that contains
// the character hash.
WTF_EXPORT_PRIVATE std::array<char, 6> integerToSixCharacterHashString(unsigned);

} // namespace WTF

using WTF::sixCharacterHashStringToInteger;
using WTF::integerToSixCharacterHashString;
