/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

#include "B3Value.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class Variable;

class JS_EXPORT_PRIVATE VariableValue final : public Value {
public:
    static bool accepts(Kind kind) { return kind == Get || kind == Set; }

    ~VariableValue() final;

    Variable* variable() const { return m_variable; }

    B3_SPECIALIZE_VALUE_FOR_NON_VARARGS_CHILDREN
    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;

    // Use this for Set.
    VariableValue(Kind, Origin, Variable*, Value*);

    // Use this for Get.
    VariableValue(Kind, Origin, Variable*);

    Variable* m_variable;
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
