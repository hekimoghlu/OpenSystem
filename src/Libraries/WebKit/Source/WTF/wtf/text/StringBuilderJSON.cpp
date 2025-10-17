/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#include "config.h"
#include <wtf/text/StringBuilderJSON.h>

#include <wtf/text/EscapedFormsForJSON.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/WTFString.h>

namespace WTF {

void StringBuilder::appendQuotedJSONString(const String& string)
{
    if (hasOverflowed())
        return;

    // Make sure we have enough buffer space to append this string for worst case without reallocating.
    // The 2 is for the '"' quotes on each end.
    // The 6 is the worst case for a single code unit that could be encoded as \uNNNN.
    CheckedInt32 stringLength = string.length();
    stringLength *= 6;
    stringLength += 2;
    if (stringLength.hasOverflowed()) {
        didOverflow();
        return;
    }

    auto stringLengthValue = stringLength.value();

    if (is8Bit() && string.is8Bit()) {
        if (auto output = extendBufferForAppending<LChar>(saturatedSum<int32_t>(m_length, stringLengthValue)); output.data()) {
            output = output.first(stringLengthValue);
            consume(output) = '"';
            appendEscapedJSONStringContent(output, string.span8());
            consume(output) = '"';
            if (!output.empty())
                shrink(m_length - output.size());
        }
    } else {
        if (auto output = extendBufferForAppendingWithUpconvert(saturatedSum<int32_t>(m_length, stringLengthValue)); output.data()) {
            output = output.first(stringLengthValue);
            consume(output) = '"';
            if (string.is8Bit())
                appendEscapedJSONStringContent(output, string.span8());
            else
                appendEscapedJSONStringContent(output, string.span16());
            consume(output) = '"';
            if (!output.empty())
                shrink(m_length - output.size());
        }
    }
}

} // namespace WTF
