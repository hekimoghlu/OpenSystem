/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

namespace PAL {

// Specifies what will happen when a character is encountered that is
// not encodable in the character set.
enum class UnencodableHandling: bool {
    // Encodes the character as an XML entity. For example, U+06DE
    // would be "&#1758;" (0x6DE = 1758 in octal).
    Entities,

    // Encodes the character as en entity as above, but escaped
    // non-alphanumeric characters. This is used in URLs.
    // For example, U+6DE would be "%26%231758%3B".
    URLEncodedEntities
};

}
