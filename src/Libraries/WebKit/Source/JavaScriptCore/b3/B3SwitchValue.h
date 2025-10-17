/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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

#include "B3CaseCollection.h"
#include "B3SwitchCase.h"
#include "B3Value.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class SwitchValue final : public Value {
public:
    static bool accepts(Kind kind) { return kind == Switch; }

    ~SwitchValue() final;

    // numCaseValues() + 1 == numSuccessors().
    unsigned numCaseValues() const { return m_values.size(); }

    // The successor for this case value is at the same index.
    int64_t caseValue(unsigned index) const { return m_values[index]; }
    
    const Vector<int64_t>& caseValues() const { return m_values; }

    CaseCollection cases(const BasicBlock* owner) const { return CaseCollection(this, owner); }
    CaseCollection cases() const { return cases(owner); }

    bool hasFallThrough(const BasicBlock*) const;
    bool hasFallThrough() const;

    BasicBlock* fallThrough(const BasicBlock* owner);

    // These two functions can be called in any order.
    void setFallThrough(BasicBlock*, const FrequentedBlock&);
    void appendCase(BasicBlock*, const SwitchCase&);
    
    JS_EXPORT_PRIVATE void setFallThrough(const FrequentedBlock&);
    JS_EXPORT_PRIVATE void appendCase(const SwitchCase&);

    void dumpSuccessors(const BasicBlock*, PrintStream&) const final;

    B3_SPECIALIZE_VALUE_FOR_FIXED_CHILDREN(1)
    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;

    static Opcode opcodeFromConstructor(Origin, Value*) { return Switch; }
    JS_EXPORT_PRIVATE SwitchValue(Origin, Value* child);

    Vector<int64_t> m_values;
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
