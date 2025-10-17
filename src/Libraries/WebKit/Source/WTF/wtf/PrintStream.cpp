/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#include <wtf/PrintStream.h>

#include <wtf/text/AtomString.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

PrintStream::PrintStream() = default;
PrintStream::~PrintStream() = default; // Force the vtable to be in this module

void PrintStream::printf(const char* format, ...)
{
    va_list argList;
    va_start(argList, format);
    vprintf(format, argList);
    va_end(argList);
}

void PrintStream::printfVariableFormat(const char* format, ...)
{
ALLOW_NONLITERAL_FORMAT_BEGIN
IGNORE_GCC_WARNINGS_BEGIN("suggest-attribute=format")
    va_list argList;
    va_start(argList, format);
    vprintf(format, argList);
    va_end(argList);
IGNORE_GCC_WARNINGS_END
ALLOW_NONLITERAL_FORMAT_END
}

void PrintStream::flush()
{
}

PrintStream& PrintStream::begin()
{
    return *this;
}

void PrintStream::end()
{
}

void printInternal(PrintStream& out, const char* string)
{
    out.printf("%s", string);
}

static void printExpectedCStringHelper(PrintStream& out, const char* type, Expected<CString, UTF8ConversionError> expectedCString)
{
    if (UNLIKELY(!expectedCString)) {
        if (expectedCString.error() == UTF8ConversionError::OutOfMemory) {
            printInternal(out, "(Out of memory while converting ");
            printInternal(out, type);
            printInternal(out, " to utf8)");
        } else {
            printInternal(out, "(failed to convert ");
            printInternal(out, type);
            printInternal(out, " to utf8)");
        }
        return;
    }
    printInternal(out, expectedCString.value());
}

void printInternal(PrintStream& out, StringView string)
{
    printExpectedCStringHelper(out, "StringView", string.tryGetUTF8());
}

void printInternal(PrintStream& out, const CString& string)
{
    printInternal(out, string.data());
}

void printInternal(PrintStream& out, const String& string)
{
    printExpectedCStringHelper(out, "String", string.tryGetUTF8());
}

void printInternal(PrintStream& out, const AtomString& string)
{
    printExpectedCStringHelper(out, "String", string.string().tryGetUTF8());
}

void printInternal(PrintStream& out, const StringImpl* string)
{
    if (!string) {
        printInternal(out, "(null StringImpl*)");
        return;
    }
    printExpectedCStringHelper(out, "StringImpl*", string->tryGetUTF8());
}

void printInternal(PrintStream& out, bool value)
{
    out.print(boolForPrinting(value));
}

void printInternal(PrintStream& out, int value)
{
    out.printf("%d", value);
}

void printInternal(PrintStream& out, unsigned value)
{
    out.printf("%u", value);
}

void printInternal(PrintStream& out, signed char value)
{
    out.printf("%d", static_cast<int>(value));
}

void printInternal(PrintStream& out, unsigned char value)
{
    out.printf("%u", static_cast<unsigned>(value));
}

void printInternal(PrintStream& out, char16_t value)
{
    out.printf("%lc", static_cast<wint_t>(value));
}

void printInternal(PrintStream& out, char32_t value)
{
    // Print each char32_t as an integer.
    out.printf("%u", static_cast<unsigned>(value));
}

void printInternal(PrintStream& out, short value)
{
    out.printf("%d", static_cast<int>(value));
}

void printInternal(PrintStream& out, unsigned short value)
{
    out.printf("%u", static_cast<unsigned>(value));
}

void printInternal(PrintStream& out, long value)
{
    out.printf("%ld", value);
}

void printInternal(PrintStream& out, unsigned long value)
{
    out.printf("%lu", value);
}

void printInternal(PrintStream& out, long long value)
{
    out.printf("%lld", value);
}

void printInternal(PrintStream& out, unsigned long long value)
{
    out.printf("%llu", value);
}

void printInternal(PrintStream& out, float value)
{
    printInternal(out, static_cast<double>(value));
}

void printInternal(PrintStream& out, double value)
{
    out.printf("%lf", value);
}

void printInternal(PrintStream& out, RawHex value)
{
#if !CPU(ADDRESS64)
    if (value.is64Bit()) {
        out.printf("0x%" PRIx64, value.u64());
        return;
    }
#endif
#if OS(WINDOWS)
    out.printf("0x%p", value.ptr());
#else
    out.printf("%p", value.ptr());
#endif
}

void printInternal(PrintStream& out, RawPointer value)
{
#if OS(WINDOWS)
    out.printf("0x%p", value.value());
#else
    out.printf("%p", value.value());
#endif
}

void printInternal(PrintStream& out, FixedWidthDouble value)
{
    out.printf("%*.*lf", value.width(), value.precision(), value.value());
}

void dumpCharacter(PrintStream& out, char value)
{
    out.printf("%c", value);
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
