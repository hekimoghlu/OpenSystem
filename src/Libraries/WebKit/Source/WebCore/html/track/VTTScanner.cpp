/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#include "VTTScanner.h"

#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

VTTScanner::VTTScanner(const String& line)
    : m_source(line)
    , m_is8Bit(line.is8Bit())
{
    if (m_is8Bit)
        m_data.characters8 = line.span8();
    else
        m_data.characters16 = line.span16();
}

bool VTTScanner::scan(char c)
{
    if (!match(c))
        return false;
    advance();
    return true;
}

bool VTTScanner::scan(std::span<const LChar> characters)
{
    auto matchLength = m_is8Bit ? m_data.characters8.size() : m_data.characters16.size();
    if (matchLength < characters.size())
        return false;
    bool matched;
    if (m_is8Bit)
        matched = equal(m_data.characters8.first(characters.size()), characters);
    else
        matched = equal(m_data.characters16.first(characters.size()), characters);
    if (matched)
        advance(characters.size());
    return matched;
}

bool VTTScanner::scanRun(const Run& run, const String& toMatch)
{
    ASSERT(run.start() == position());
    ASSERT(run.start() <= end());
    ASSERT(run.end() >= run.start());
    ASSERT(run.end() <= end());
    size_t matchLength = run.length();
    if (toMatch.length() > matchLength)
        return false;
    bool matched;
    if (m_is8Bit)
        matched = equal(toMatch.impl(), m_data.characters8.first(matchLength));
    else
        matched = equal(toMatch.impl(), m_data.characters16.first(matchLength));
    if (matched)
        advance(run.length());
    return matched;
}

void VTTScanner::skipRun(const Run& run)
{
    ASSERT(run.start() <= end());
    ASSERT(run.end() >= run.start());
    ASSERT(run.end() <= end());
    seekTo(run.end());
}

String VTTScanner::extractString(const Run& run)
{
    ASSERT(run.start() == position());
    ASSERT(run.start() <= end());
    ASSERT(run.end() >= run.start());
    ASSERT(run.end() <= end());
    String s;
    if (m_is8Bit)
        s = run.span8();
    else
        s = run.span16();
    advance(run.length());
    return s;
}

String VTTScanner::restOfInputAsString()
{
    return extractString(m_is8Bit ? Run { m_data.characters8 } : Run { m_data.characters16 });
}

unsigned VTTScanner::scanDigits(unsigned& number)
{
    Run runOfDigits = collectWhile<isASCIIDigit>();
    if (runOfDigits.isEmpty()) {
        number = 0;
        return 0;
    }

    StringView string;
    unsigned numDigits = runOfDigits.length();
    if (m_is8Bit)
        string = m_data.characters8.first(numDigits);
    else
        string = m_data.characters16.first(numDigits);

    // Since these are ASCII digits, the only failure mode is overflow, so use the maximum unsigned value.
    number = parseInteger<unsigned>(string).value_or(std::numeric_limits<unsigned>::max());

    // Consume the digits.
    advance(runOfDigits.length());
    return numDigits;
}

bool VTTScanner::scanFloat(float& number, bool* isNegative)
{
    bool negative = scan('-');
    Run integerRun = collectWhile<isASCIIDigit>();

    advance(integerRun.length());
    Run decimalRun = createRun(position(), position());
    if (scan('.')) {
        decimalRun = collectWhile<isASCIIDigit>();
        advance(decimalRun.length());
    }

    // At least one digit required.
    if (integerRun.isEmpty() && decimalRun.isEmpty()) {
        // Restore to starting position.
        seekTo(integerRun.start());
        return false;
    }

    Run floatRun = createRun(integerRun.start(), position());
    bool validNumber;
    if (m_is8Bit)
        number = charactersToFloat(floatRun.span8(), &validNumber);
    else
        number = charactersToFloat(floatRun.span16(), &validNumber);

    if (!validNumber)
        number = std::numeric_limits<float>::max();
    else if (negative)
        number = -number;

    if (isNegative)
        *isNegative = negative;

    return true;
}

auto VTTScanner::createRun(Position start, Position end) const -> Run
{
    if (m_is8Bit) {
        auto span8 = m_source.span8();
        auto* start8 = static_cast<const LChar*>(start);
        auto* end8 = static_cast<const LChar*>(end);
        RELEASE_ASSERT(start8 >= span8.data());
        return Run { span8.subspan(start8 - span8.data(), end8 - start8) };
    }
    auto span16 = m_source.span16();
    auto* start16 = static_cast<const UChar*>(start);
    auto* end16 = static_cast<const UChar*>(end);
    RELEASE_ASSERT(start16 >= span16.data());
    return Run { span16.subspan(start16 - span16.data(), end16 - start16) };
}

}
