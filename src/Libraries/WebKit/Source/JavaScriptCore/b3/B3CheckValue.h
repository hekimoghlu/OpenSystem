/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

#include "B3StackmapValue.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class CheckValue final : public StackmapValue {
public:
    static bool accepts(Kind kind)
    {
        switch (kind.opcode()) {
        case CheckAdd:
        case CheckSub:
        case CheckMul:
        case Check:
            return true;
        default:
            return false;
        }
    }

    ~CheckValue() final;

    void convertToAdd();

    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_VARARGS_CHILDREN
    
private:
    friend class Procedure;
    friend class Value;

    // Use this form for CheckAdd, CheckSub, and CheckMul.
    JS_EXPORT_PRIVATE CheckValue(Kind, Origin, Value* left, Value* right);

    // Use this form for Check.
    JS_EXPORT_PRIVATE CheckValue(Kind, Origin, Value* predicate);
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
