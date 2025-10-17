/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

#include <wtf/Forward.h>
#include <wtf/RobinHoodHashSet.h>

namespace WebCore {

// This setting is used to define which types of fonts are trusted to be downloaded and loaded into the system
// using the default, possibly-unsafe font parser.
// Any: any font binary will be downloaded, no checks will be done during load.
// Restricted: any font binary will be downloaded but just binaries listed by WebKit are trusted to load through the system font parser. A not allowed binary will be deleted after the check is done.
// SafeFontParser: any font binary will be downloaded. Binaries listed by WebKit are trusted to load through the safe font parser.
// None: No font binary will be downloaded or loaded.
enum class DownloadableBinaryFontTrustedTypes : uint8_t {
    Any,
    Restricted,
    SafeFontParser,
    None
};

// Identifies the policy to respect for loading font binaries.
// Deny: do not load the font binary.
// LoadWithSystemFontParser: font can be loaded with the possibly-unsafe system font parser.
// LoadWithSafeFontParser: font can be loaded with a safe font parser (with no fallback on the system font parser).
enum class FontParsingPolicy : uint8_t {
    Deny,
    LoadWithSystemFontParser,
    LoadWithSafeFontParser,
};

FontParsingPolicy fontBinaryParsingPolicy(std::span<const uint8_t>, DownloadableBinaryFontTrustedTypes);

} // namespace WebCore
