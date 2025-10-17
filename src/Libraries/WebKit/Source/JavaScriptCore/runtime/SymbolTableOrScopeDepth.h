/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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

#include "VirtualRegister.h"
#include <wtf/PrintStream.h>

namespace JSC {

class SymbolTableOrScopeDepth {
public:
    SymbolTableOrScopeDepth() = default;

    static SymbolTableOrScopeDepth symbolTable(VirtualRegister reg)
    {
        ASSERT(reg.isConstant());
        return SymbolTableOrScopeDepth(reg.offset() - FirstConstantRegisterIndex);
    }

    static SymbolTableOrScopeDepth scopeDepth(unsigned scopeDepth)
    {
        return SymbolTableOrScopeDepth(scopeDepth);
    }

    static SymbolTableOrScopeDepth raw(unsigned value)
    {
        return SymbolTableOrScopeDepth(value);
    }

    VirtualRegister symbolTable() const
    {
        return VirtualRegister(m_raw + FirstConstantRegisterIndex);
    }

    unsigned scopeDepth() const
    {
        return m_raw;
    }

    unsigned raw() const { return m_raw; }

    void dump(PrintStream& out) const
    {
        out.print(m_raw);
    }

private:
    SymbolTableOrScopeDepth(unsigned value)
        : m_raw(value)
    { }

    unsigned m_raw { 0 };
};

} // namespace JSC
