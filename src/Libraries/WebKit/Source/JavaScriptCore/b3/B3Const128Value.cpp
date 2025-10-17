/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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

#if ENABLE(B3_JIT)
#include "B3Const128Value.h"

#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

Const128Value::~Const128Value() = default;

Value* Const128Value::vectorAndConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasV128())
        return nullptr;
    v128_t result = vectorAnd(m_value, other->asV128());
    return proc.add<Const128Value>(origin(), result);
}

Value* Const128Value::vectorOrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasV128())
        return nullptr;
    v128_t result = vectorOr(m_value, other->asV128());
    return proc.add<Const128Value>(origin(), result);
}

Value* Const128Value::vectorXorConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasV128())
        return nullptr;
    v128_t result = vectorXor(m_value, other->asV128());
    return proc.add<Const128Value>(origin(), result);
}

void Const128Value::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma, m_value.u64x2[0], comma, m_value.u64x2[1]);
}

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
