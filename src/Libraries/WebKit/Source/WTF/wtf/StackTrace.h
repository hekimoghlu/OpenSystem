/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include <optional>
#include <span>
#include <wtf/Forward.h>
#include <wtf/SystemFree.h>

#if HAVE(BACKTRACE_SYMBOLS) || HAVE(BACKTRACE)
#include <execinfo.h>
#endif

#if USE(LIBBACKTRACE)
#include <backtrace.h>
#endif

#if HAVE(DLADDR)
#include <cxxabi.h>
#include <dlfcn.h>
#endif

#if OS(WINDOWS)
#include <windows.h>
#include <wtf/win/DbgHelperWin.h>
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

class PrintStream;

#if USE(LIBBACKTRACE)
WTF_EXPORT_PRIVATE char** symbolize(void* const*, int);
#endif

class StackTrace {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WTF_EXPORT_PRIVATE NEVER_INLINE static std::unique_ptr<StackTrace> captureStackTrace(size_t maxFrames, size_t framesToSkip = 0);

    std::span<void* const> stack() const
    {
        return std::span<void* const> { m_stack + m_initialFrame, m_size };
    }

    void dump(PrintStream&) const;
    void forEachFrame(NOESCAPE const std::invocable<int, void*, const char*> auto&) const;
    WTF_EXPORT_PRIVATE String toString() const;

private:
    StackTrace(size_t size, size_t initialFrame)
        : m_size(size)
        , m_initialFrame(initialFrame)
    {
    }

    size_t m_size;
    size_t m_initialFrame;
    void* m_stack[1];
};

class StackTraceSymbolResolver {
public:
    StackTraceSymbolResolver(std::span<void* const> stack)
        : m_stack(stack)
    {
    }
    StackTraceSymbolResolver(const StackTrace& stack)
        : m_stack(stack.stack())
    {
    }

    class DemangleEntry {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        friend class StackTraceSymbolResolver;
        const char* mangledName() const { return m_mangledName; }
        const char* demangledName() const { return m_demangledName.get(); }

    private:
        DemangleEntry(const char* mangledName, const char* demangledName)
            : m_mangledName(mangledName)
            , m_demangledName(demangledName)
        { }

        const char* m_mangledName { nullptr };
        std::unique_ptr<const char[], SystemFree<const char[]>> m_demangledName;
    };

    WTF_EXPORT_PRIVATE static std::optional<DemangleEntry> demangle(void*);

    void forEach(NOESCAPE const std::invocable<int, void*, const char*> auto& functor) const
    {
#if USE(LIBBACKTRACE)
        char** symbols = symbolize(m_stack.data(), m_stack.size());
        if (!symbols)
            return;
#elif HAVE(BACKTRACE_SYMBOLS)
        char** symbols = backtrace_symbols(m_stack.data(), m_stack.size());
        if (!symbols)
            return;
#elif OS(WINDOWS)
        HANDLE hProc = GetCurrentProcess();
        uint8_t symbolData[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)] = { 0 };
        auto symbolInfo = reinterpret_cast<SYMBOL_INFO*>(symbolData);

        symbolInfo->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo->MaxNameLen = MAX_SYM_NAME;
#endif
        for (size_t i = 0; i < m_stack.size(); ++i) {
            const char* name = nullptr;
            auto demangled = demangle(m_stack[i]);
            if (demangled)
                name = demangled->demangledName() ? demangled->demangledName() : demangled->mangledName();
#if HAVE(BACKTRACE_SYMBOLS)
            if (!name || !strcmp(name, "<redacted>"))
                name = symbols[i];
#elif OS(WINDOWS)
            if (!name && DbgHelper::SymFromAddress(hProc, reinterpret_cast<DWORD64>(m_stack[i]), nullptr, symbolInfo))
                name = symbolInfo->Name;
#endif
            functor(i + 1, m_stack[i], name);
        }

#if USE(LIBBACKTRACE)
        for (size_t i = 0; i < m_stack.size(); ++i)
            free(symbols[i]);
        free(symbols);
#elif HAVE(BACKTRACE_SYMBOLS)
        free(symbols);
#endif
    }
private:
    std::span<void* const> m_stack;
};

class StackTracePrinter {
public:
    StackTracePrinter(std::span<void* const> stack, const char* prefix = "")
        : m_stack(stack)
        , m_prefix(prefix)
    {
    }

    StackTracePrinter(const StackTrace& stack, const char* prefix = "")
        : m_stack(stack.stack())
        , m_prefix(prefix)
    {
    }

    WTF_EXPORT_PRIVATE void dump(PrintStream&) const;

private:
    const std::span<void* const> m_stack;
    const char* const m_prefix;
};

inline void StackTrace::dump(PrintStream& out) const
{
    StackTracePrinter { *this }.dump(out);
}

void StackTrace::forEachFrame(NOESCAPE const std::invocable<int, void*, const char*> auto& functor) const
{
    StackTraceSymbolResolver { *this }.forEach(functor);
}

} // namespace WTF

using WTF::StackTrace;
using WTF::StackTraceSymbolResolver;
using WTF::StackTracePrinter;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
