/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

#include "CPU.h"

#include <wtf/PrintStream.h>
#include <wtf/StringPrintStream.h>
#include <wtf/Vector.h>

namespace JSC {

namespace Probe {
class Context;
} // namespace Probe

namespace Printer {

struct Context;

union Data {
    Data()
    {
        const intptr_t uninitialized = 0xdeadb0d0;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        memcpy(&buffer, &uninitialized, sizeof(uninitialized));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }
    Data(uintptr_t value)
        : Data(&value, sizeof(value))
    { }
    Data(const void* pointer)
        : Data(&pointer, sizeof(pointer))
    { }
    Data(void* src, size_t size)
    {
        RELEASE_ASSERT(size <= sizeof(buffer));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        memcpy(&buffer, src, size);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }

    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    T as() const
    {
        return static_cast<T>(value);
    }

    template<typename T, typename = typename std::enable_if<std::is_pointer<T>::value>::type>
    const T as(int = 0) const
    {
        return reinterpret_cast<const T>(pointer);
    }

    template<typename T, typename = typename std::enable_if<!std::is_integral<T>::value && !std::is_pointer<T>::value>::type>
    const T& as() const
    {
        static_assert(sizeof(T) <= sizeof(buffer), "size is not sane");
        return *reinterpret_cast<const T*>(&buffer);
    }

    uintptr_t value;
    const void* pointer;
#if USE(JSVALUE64)
    UCPURegister buffer[4];
#elif USE(JSVALUE32_64)
    UCPURegister buffer[6];
#endif
};

struct Context {
    Context(Probe::Context& probeContext, Data& data)
        : probeContext(probeContext)
        , data(data)
    { }

    Probe::Context& probeContext;
    Data& data;
};

typedef void (*Callback)(PrintStream&, Context&);

struct PrintRecord {
    PrintRecord(Data data, Callback printer)
        : data(data)
        , printer(printer)
    { }

    PrintRecord(Callback printer)
        : printer(printer)
    { }

    template<template<class> class Printer, typename T>
    PrintRecord(const Printer<T>& other)
    {
        static_assert(std::is_base_of<PrintRecord, Printer<T>>::value, "Printer should extend PrintRecord");
        static_assert(sizeof(PrintRecord) == sizeof(Printer<T>), "Printer should be the same size as PrintRecord");
        data = other.data;
        printer = other.printer;
    }

    Data data;
    Callback printer;

protected:
    PrintRecord() { }
};

template<typename T> struct Printer;

typedef Vector<PrintRecord> PrintRecordList;

inline void appendPrinter(PrintRecordList&) { }

template<typename First, typename... Arguments>
inline void appendPrinter(PrintRecordList& printRecordList, First first, Arguments&&... others)
{
    printRecordList.append(Printer<First>(first));
    appendPrinter(printRecordList, std::forward<Arguments>(others)...);
}

template<typename... Arguments>
inline PrintRecordList* makePrintRecordList(Arguments&&... arguments)
{
    // FIXME: the current implementation intentionally leaks the PrintRecordList.
    // We may want to fix this in the future if we want to use the print mechanism
    // in tests that may compile a lot of prints.
    // https://bugs.webkit.org/show_bug.cgi?id=171123
    auto printRecordList = new PrintRecordList();
    appendPrinter(*printRecordList, std::forward<Arguments>(arguments)...);
    return printRecordList;
}

// Some utility functions for specializing printers.

void printConstCharString(PrintStream&, Context&);
void printIntptr(PrintStream&, Context&);
void printUintptr(PrintStream&, Context&);
void printPointer(PrintStream&, Context&);

void setPrinter(PrintRecord&, CString&&);

// Specialized printers.

template<>
struct Printer<const char*> : public PrintRecord {
    Printer(const char* str)
        : PrintRecord(str, printConstCharString)
    { }
};

template<>
struct Printer<char*> : public Printer<const char*> {
    Printer(char* str)
        : Printer<const char*>(str)
    { }
};

template<>
struct Printer<RawPointer> : public PrintRecord {
    Printer(RawPointer rawPointer)
        : PrintRecord(rawPointer.value(), printPointer)
    { }
};

template<typename T>
std::enable_if_t<std::is_integral_v<T>>
setPrinter(PrintRecord& record, T value, intptr_t = 0)
{
    record.data.value = static_cast<uintptr_t>(value);
    if constexpr (std::numeric_limits<T>::is_signed)
        record.printer = printIntptr;
    else
        record.printer = printUintptr;
}

template<typename T>
struct Printer : public PrintRecord {
    Printer(T value)
    {
        setPrinter(*this, value);
    }
};

} // namespace Printer

} // namespace JSC
