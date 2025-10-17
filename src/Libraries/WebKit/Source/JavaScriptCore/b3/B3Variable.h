/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#if ENABLE(B3_JIT)

#include "B3Bank.h"
#include "B3SparseCollection.h"
#include "B3Type.h"
#include "B3Width.h"
#include <wtf/Noncopyable.h>
#include <wtf/PrintStream.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 {

class Procedure;

class Variable {
    WTF_MAKE_NONCOPYABLE(Variable);
    WTF_MAKE_TZONE_ALLOCATED(Variable);

public:
    ~Variable();

    Type type() const { return m_type; }
    Width width() const { return widthForType(type()); }
    Bank bank() const { return bankForType(type()); }
    unsigned index() const { return m_index; }

    void dump(PrintStream&) const;
    void deepDump(PrintStream&) const;

private:
    friend class Procedure;
    friend class SparseCollection<Variable>;

    Variable(Type);
    
    unsigned m_index;
    Type m_type;
};

class DeepVariableDump {
public:
    DeepVariableDump(const Variable* variable)
        : m_variable(variable)
    {
    }

    void dump(PrintStream& out) const
    {
        if (m_variable)
            m_variable->deepDump(out);
        else
            out.print("<null>");
    }

private:
    const Variable* m_variable;
};

inline DeepVariableDump deepDump(const Variable* variable)
{
    return DeepVariableDump(variable);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
