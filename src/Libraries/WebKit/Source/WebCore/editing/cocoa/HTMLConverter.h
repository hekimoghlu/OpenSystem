/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
#import "AttributedString.h"
#import "SimpleRange.h"

namespace WebCore {

enum class IgnoreUserSelectNone : bool;
enum class TextIteratorBehavior : uint16_t;

WEBCORE_EXPORT AttributedString attributedString(const SimpleRange&, IgnoreUserSelectNone);

// This alternate implementation of HTML conversion doesn't handle as many advanced features,
// such as tables, and doesn't produce document attributes, but it does use TextIterator so
// text offsets will exactly match plain text and other editing machinery.
// FIXME: This function and the one above should be merged.

enum class IncludedElement : uint8_t {
    Images = 1 << 0,
    Attachments = 1 << 1,
    PreservedContent = 1 << 2,
    NonRenderedContent = 1 << 3,
};

WEBCORE_EXPORT AttributedString editingAttributedString(const SimpleRange&, OptionSet<IncludedElement> = { IncludedElement::Images });
WEBCORE_EXPORT AttributedString editingAttributedStringReplacingNoBreakSpace(const SimpleRange&, OptionSet<TextIteratorBehavior>, OptionSet<IncludedElement>);

} // namespace WebCore
