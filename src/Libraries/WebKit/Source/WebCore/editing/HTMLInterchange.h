/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

class Text;

constexpr auto AppleInterchangeNewline = "Apple-interchange-newline"_s;
constexpr auto AppleConvertedSpace = "Apple-converted-space"_s;
constexpr auto WebKitMSOListQuirksStyle = "WebKit-mso-list-quirks-style"_s;

constexpr auto ApplePasteAsQuotation = "Apple-paste-as-quotation"_s;
constexpr auto AppleStyleSpanClass = "Apple-style-span"_s;
constexpr auto AppleTabSpanClass = "Apple-tab-span"_s;

// Controls whether a special BR which is removed upon paste in ReplaceSelectionCommand needs to be inserted
// and making sequence of spaces not collapsible by inserting non-breaking spaces.
// See https://trac.webkit.org/r8087 and https://trac.webkit.org/r8096.
enum class AnnotateForInterchange : bool { No, Yes };

String convertHTMLTextToInterchangeFormat(const String&, const Text*);

} // namespace WebCore
