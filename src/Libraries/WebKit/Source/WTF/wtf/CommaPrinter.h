/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#include <wtf/text/ASCIILiteral.h>

namespace WTF {

class CommaPrinter final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    CommaPrinter(ASCIILiteral comma = ", "_s, ASCIILiteral start = ""_s)
        : m_comma(comma)
        , m_start(start)
        , m_didPrint(false)
    {
    }
    
    void dump(PrintStream& out) const
    {
        if (!m_didPrint) {
            out.print(m_start);
            m_didPrint = true;
            return;
        }
        
        out.print(m_comma);
    }
    
    bool didPrint() const { return m_didPrint; }
    
private:
    ASCIILiteral m_comma;
    ASCIILiteral m_start;
    mutable bool m_didPrint;
};

} // namespace WTF

using WTF::CommaPrinter;
