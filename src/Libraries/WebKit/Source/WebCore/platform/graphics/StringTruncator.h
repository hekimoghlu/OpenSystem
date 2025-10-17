/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#ifndef StringTruncator_h
#define StringTruncator_h

#include <wtf/Forward.h>

namespace WebCore {
    
class FontCascade;

class StringTruncator {
public:
    WEBCORE_EXPORT static String centerTruncate(const String&, float maxWidth, const FontCascade&);
    WEBCORE_EXPORT static String rightTruncate(const String&, float maxWidth, const FontCascade&);

    WEBCORE_EXPORT static String centerTruncate(const String&, float maxWidth, const FontCascade&, float& resultWidth, bool shouldInsertEllipsis = true, float customTruncationElementWidth = 0);
    WEBCORE_EXPORT static String rightTruncate(const String&, float maxWidth, const FontCascade&, float& resultWidth, bool shouldInsertEllipsis = true, float customTruncationElementWidth = 0);
    WEBCORE_EXPORT static String leftTruncate(const String&, float maxWidth, const FontCascade&, float& resultWidth, bool shouldInsertEllipsis = true, float customTruncationElementWidth = 0);
    WEBCORE_EXPORT static String rightClipToCharacter(const String&, float maxWidth, const FontCascade&, float& resultWidth, bool shouldInsertEllipsis = true, float customTruncationElementWidth = 0);
    WEBCORE_EXPORT static String rightClipToWord(const String&, float maxWidth, const FontCascade&, float& resultWidth, bool shouldInsertEllipsis = true, float customTruncationElementWidth = 0, bool alwaysTruncate = false);

    WEBCORE_EXPORT static float width(const String&, const FontCascade&);
};
    
} // namespace WebCore

#endif // !defined(StringTruncator_h)
