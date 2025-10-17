/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include <memory>
#include <wtf/Forward.h>

#if PLATFORM(COCOA)
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace PAL {

class TextCodec;
class TextEncoding;

// Use TextResourceDecoder::decode to decode resources, since it handles BOMs.
// Use TextEncoding::encode to encode, since it takes care of normalization.
PAL_EXPORT std::unique_ptr<TextCodec> newTextCodec(const TextEncoding&);

// Only TextEncoding should use the following functions directly.
ASCIILiteral atomCanonicalTextEncodingName(ASCIILiteral alias);
ASCIILiteral atomCanonicalTextEncodingName(StringView);
bool noExtendedTextEncodingNameUsed();
bool isJapaneseEncoding(ASCIILiteral canonicalEncodingName);
bool shouldShowBackslashAsCurrencySymbolIn(ASCIILiteral canonicalEncodingName);

PAL_EXPORT String defaultTextEncodingNameForSystemLanguage();

#if PLATFORM(COCOA)
PAL_EXPORT CFStringEncoding webDefaultCFStringEncoding();
#endif

} // namespace PAL
