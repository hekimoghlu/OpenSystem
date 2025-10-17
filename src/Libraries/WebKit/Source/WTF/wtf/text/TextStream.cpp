/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#include <wtf/text/TextStream.h>

#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WTF {

static constexpr size_t printBufferSize = 100; // large enough for any integer or floating point value in string format, including trailing null character

static inline bool hasFractions(double val)
{
    static constexpr double s_epsilon = 0.0001;
    int ival = static_cast<int>(val);
    double dval = static_cast<double>(ival);
    return std::abs(val - dval) > s_epsilon;
}

TextStream& TextStream::operator<<(bool b)
{
    return *this << (b ? "1" : "0");
}

TextStream& TextStream::operator<<(char c)
{
    m_text.append(c);
    return *this;
}

TextStream& TextStream::operator<<(int i)
{
    m_text.append(i);
    return *this;
}

TextStream& TextStream::operator<<(unsigned i)
{
    m_text.append(i);
    return *this;
}

TextStream& TextStream::operator<<(long i)
{
    m_text.append(i);
    return *this;
}

TextStream& TextStream::operator<<(unsigned long i)
{
    m_text.append(i);
    return *this;
}

TextStream& TextStream::operator<<(long long i)
{
    m_text.append(i);
    return *this;
}

TextStream& TextStream::operator<<(unsigned long long i)
{
    m_text.append(i);
    return *this;
}

TextStream& TextStream::operator<<(float f)
{
    if (m_formattingFlags & Formatting::NumberRespectingIntegers)
        return *this << FormatNumberRespectingIntegers(f);

    m_text.append(FormattedNumber::fixedWidth(f, 2));
    return *this;
}

TextStream& TextStream::operator<<(double d)
{
    if (m_formattingFlags & Formatting::NumberRespectingIntegers)
        return *this << FormatNumberRespectingIntegers(d);

    m_text.append(FormattedNumber::fixedWidth(d, 2));
    return *this;
}

TextStream& TextStream::operator<<(const char* string)
{
    m_text.append(unsafeSpan(string));
    return *this;
}

TextStream& TextStream::operator<<(const void* p)
{
    char buffer[printBufferSize];
    SAFE_SPRINTF(std::span { buffer }, "%p", p);
    return *this << buffer;
}

TextStream& TextStream::operator<<(const AtomString& string)
{
    m_text.append(string);
    return *this;
}

TextStream& TextStream::operator<<(const CString& string)
{
    m_text.append(string);
    return *this;
}

TextStream& TextStream::operator<<(const String& string)
{
    m_text.append(string);
    return *this;
}

TextStream& TextStream::operator<<(ASCIILiteral string)
{
    m_text.append(string);
    return *this;
}

TextStream& TextStream::operator<<(StringView string)
{
    m_text.append(string);
    return *this;
}

TextStream& TextStream::operator<<(const HexNumberBuffer& buffer)
{
    m_text.append(buffer);
    return *this;
}

TextStream& TextStream::operator<<(const FormattedCSSNumber& number)
{
    m_text.append(number);
    return *this;
}

TextStream& TextStream::operator<<(const FormatNumberRespectingIntegers& numberToFormat)
{
    if (hasFractions(numberToFormat.value)) {
        m_text.append(FormattedNumber::fixedWidth(numberToFormat.value, 2));
        return *this;
    }

    m_text.append(static_cast<int>(numberToFormat.value));
    return *this;
}

String TextStream::release()
{
    String result = m_text.toString();
    m_text.clear();
    return result;
}

void TextStream::startGroup()
{
    TextStream& ts = *this;

    if (m_multiLineMode) {
        ts << "\n";
        ts.writeIndent();
        ts << "(";
        ts.increaseIndent();
    } else
        ts << " (";
}

void TextStream::endGroup()
{
    TextStream& ts = *this;
    ts << ")";
    if (m_multiLineMode)
        ts.decreaseIndent();
}

void TextStream::nextLine()
{
    TextStream& ts = *this;
    if (m_multiLineMode) {
        ts << "\n";
        ts.writeIndent();
    } else
        ts << " ";
}

void TextStream::writeIndent()
{
    if (m_multiLineMode)
        WTF::writeIndent(*this, m_indent);
}

void writeIndent(TextStream& ts, int indent)
{
    for (int i = 0; i < indent; ++i)
        ts << "  ";
}

} // namespace WTF
