/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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

#if ENABLE(DFG_JIT)

#include "DFGMinifiedGraph.h"
#include "DFGVariableEvent.h"
#include "Operands.h"
#include "ValueRecovery.h"
#include <wtf/Vector.h>

namespace JSC { namespace DFG {

struct UndefinedOperandSpan {
    unsigned firstIndex;
    int minOffset;
    unsigned numberOfRegisters;
};

class VariableEventStream {
public:
    VariableEventStream() = default;
    VariableEventStream(Vector<VariableEvent>&& stream)
        : m_stream(WTFMove(stream))
    {
    }

    unsigned reconstruct(CodeBlock*, CodeOrigin, MinifiedGraph&, unsigned index, Operands<ValueRecovery>&) const;
    unsigned reconstruct(CodeBlock*, CodeOrigin, MinifiedGraph&, unsigned index, Operands<ValueRecovery>&, Vector<UndefinedOperandSpan>*) const;

private:
    enum class ReconstructionStyle {
        Combined,
        Separated
    };
    template<ReconstructionStyle style>
    unsigned reconstruct(
        CodeBlock*, CodeOrigin, MinifiedGraph&,
        unsigned index, Operands<ValueRecovery>&, Vector<UndefinedOperandSpan>*) const;

    FixedVector<VariableEvent> m_stream;
};

class VariableEventStreamBuilder {
    WTF_MAKE_NONCOPYABLE(VariableEventStreamBuilder);
public:
    static constexpr bool verbose = false;

    VariableEventStreamBuilder() = default;

    void appendAndLog(const VariableEvent& event)
    {
        if (verbose)
            logEvent(event);
        m_stream.append(event);
    }

    unsigned size() const { return m_stream.size(); }

    Vector<VariableEvent> finalize() { return WTFMove(m_stream); }

private:
    void logEvent(const VariableEvent&);

    Vector<VariableEvent> m_stream;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
