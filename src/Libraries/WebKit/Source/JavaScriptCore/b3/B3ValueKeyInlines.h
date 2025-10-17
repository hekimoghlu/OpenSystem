/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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

#include "B3Procedure.h"
#include "B3Value.h"
#include "B3ValueKey.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

inline ValueKey::ValueKey(Kind kind, Type type, Value* child)
    : m_kind(kind)
    , m_type(type)
{
    u.indices[0] = child->index();
}

inline ValueKey::ValueKey(Kind kind, Type type, Value* left, Value* right)
    : m_kind(kind)
    , m_type(type)
{
    u.indices[0] = left->index();
    u.indices[1] = right->index();
}

inline ValueKey::ValueKey(Kind kind, Type type, Value* a, Value* b, Value* c)
    : m_kind(kind)
    , m_type(type)
{
    u.indices[0] = a->index();
    u.indices[1] = b->index();
    u.indices[2] = c->index();
}

inline ValueKey::ValueKey(Kind kind, Type type, SIMDInfo simdInfo, Value* a)
    : m_simdInfo(simdInfo)
    , m_kind(kind)
    , m_type(type)
{
    u.indices[0] = a ? a->index() : UINT32_MAX;
}

inline ValueKey::ValueKey(Kind kind, Type type, SIMDInfo simdInfo, Value* a, Value* b)
    : m_simdInfo(simdInfo)
    , m_kind(kind)
    , m_type(type)
{
    u.indices[0] = a ? a->index() : UINT32_MAX;
    u.indices[1] = b ? b->index() : UINT32_MAX;
}

inline ValueKey::ValueKey(Kind kind, Type type, SIMDInfo simdInfo, Value* a, Value* b, Value* c)
    : m_simdInfo(simdInfo)
    , m_kind(kind)
    , m_type(type)
{
    u.indices[0] = a ? a->index() : UINT32_MAX;
    u.indices[1] = b ? b->index() : UINT32_MAX;
    u.indices[2] = c ? c->index() : UINT32_MAX;
}

inline ValueKey::ValueKey(Kind kind, Type type, SIMDInfo simdInfo, Value* a, uint8_t immediate)
    : m_simdInfo(simdInfo)
    , m_kind(kind)
    , m_type(type)
{
    u.indices[0] = a ? a->index() : UINT32_MAX;
    u.indices[1] = immediate;
}

inline ValueKey::ValueKey(Kind kind, Type type, SIMDInfo simdInfo, Value* a, Value* b, uint8_t immediate)
    : m_simdInfo(simdInfo)
    , m_kind(kind)
    , m_type(type)
{
    u.indices[0] = a ? a->index() : UINT32_MAX;
    u.indices[1] = b ? b->index() : UINT32_MAX;
    u.indices[2] = immediate;
}

inline Value* ValueKey::child(Procedure& proc, unsigned index) const
{
    return proc.values()[index];
}

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
