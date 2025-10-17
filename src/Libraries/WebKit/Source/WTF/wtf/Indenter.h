/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include <wtf/text/WTFString.h>

namespace WTF {

class Indenter {
    WTF_MAKE_FAST_ALLOCATED;
public:
    Indenter(unsigned count = 0, String string = "  "_s)
        : m_count(count)
        , m_string(string)
    { }

    Indenter(const Indenter& other)
        : m_count(other.m_count)
        , m_string(other.m_string)
    { }

    void dump(PrintStream& out) const
    {
        unsigned levels = m_count;
        while (levels--)
            out.print(m_string);
    }

    unsigned operator++() { return ++m_count; }
    unsigned operator++(int) { return m_count++; }
    unsigned operator--() { return --m_count; }
    unsigned operator--(int) { return m_count--; }

private:
    unsigned m_count;
    String m_string;
};

} // namespace WTF

using WTF::Indenter;
