/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

#include <wtf/PrintStream.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace WTF {

class StringPrintStream final : public PrintStream {
public:
    WTF_EXPORT_PRIVATE StringPrintStream();
    WTF_EXPORT_PRIVATE ~StringPrintStream() final;
    
    WTF_EXPORT_PRIVATE void vprintf(const char* format, va_list) final WTF_ATTRIBUTE_PRINTF(2, 0);

    size_t length() const { return m_length; }
    
    WTF_EXPORT_PRIVATE CString toCString() const;
    WTF_EXPORT_PRIVATE Expected<String, UTF8ConversionError> tryToString() const;
    WTF_EXPORT_PRIVATE String toString() const;
    WTF_EXPORT_PRIVATE String toStringWithLatin1Fallback() const;
    WTF_EXPORT_PRIVATE void reset();
    
private:
    void increaseSize(size_t);
    
    std::array<char, 128> m_inlineBuffer;
    std::span<char> m_buffer;
    size_t m_length { 0 };
};

// Stringify any type T that has a WTF::printInternal(PrintStream&, const T&)

template<typename... Types>
CString toCString(const Types&... values)
{
    StringPrintStream stream;
    stream.print(values...);
    return stream.toCString();
}

template<typename... Types>
String toString(const Types&... values)
{
    StringPrintStream stream;
    stream.print(values...);
    return stream.toString();
}

} // namespace WTF

using WTF::StringPrintStream;
using WTF::toCString;
using WTF::toString;
