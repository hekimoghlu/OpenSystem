/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

#include <wtf/StringPrintStream.h>

namespace WTF {

// This class is intended for when you want to easily buffer and print a bunch of information
// at the end of some scope/function.
class ScopedPrintStream final : public PrintStream {
public:
    ScopedPrintStream(PrintStream& out = WTF::dataFile())
        : m_out(out)
    { }

    ~ScopedPrintStream() final
    {
        m_out.print(m_buffer.toCString());
        m_out.flush();
    }

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    void vprintf(const char* format, va_list argList) final WTF_ATTRIBUTE_PRINTF(2, 0)
    {
        m_buffer.vprintf(format, argList);
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    void reset() { m_buffer.reset(); }

private:
    StringPrintStream m_buffer;
    PrintStream& m_out;
};

} // namespace WTF

using WTF::ScopedPrintStream;
