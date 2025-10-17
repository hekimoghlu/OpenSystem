/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#include "config.h"
#include "B3CheckValue.h"

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

CheckValue::~CheckValue() = default;

void CheckValue::convertToAdd()
{
    RELEASE_ASSERT(opcode() == CheckAdd || opcode() == CheckSub || opcode() == CheckMul);
    m_kind = CheckAdd;
}

// Use this form for CheckAdd, CheckSub, and CheckMul.
CheckValue::CheckValue(Kind kind, Origin origin, Value* left, Value* right)
    : StackmapValue(CheckedOpcode, kind, left->type(), origin)
{
    ASSERT(type().isInt());
    ASSERT(left->type() == right->type());
    ASSERT(kind == CheckAdd || kind == CheckSub || kind == CheckMul);
    append(ConstrainedValue(left, ValueRep::WarmAny));
    append(ConstrainedValue(right, ValueRep::WarmAny));
}

// Use this form for Check.
CheckValue::CheckValue(Kind kind, Origin origin, Value* predicate)
    : StackmapValue(CheckedOpcode, kind, Void, origin)
{
    ASSERT(kind == Check);
    append(ConstrainedValue(predicate, ValueRep::WarmAny));
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
