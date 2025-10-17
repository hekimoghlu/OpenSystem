/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#include "Structure.h"

namespace JSC { namespace DFG {

class Graph;

class RegisteredStructure {
public:
    RegisteredStructure() = default;

    ALWAYS_INLINE Structure* get() const { return m_structure; }
    Structure* operator->() const { return get(); }

    friend bool operator==(const RegisteredStructure&, const RegisteredStructure&) = default;

    explicit operator bool() const
    {
        return !!get();
    }

private:
    friend class Graph;

    RegisteredStructure(Structure* structure)
        : m_structure(structure)
    {
        ASSERT(structure);
    }

    static RegisteredStructure createPrivate(Structure* structure)
    {
        return RegisteredStructure(structure);
    }

    Structure* m_structure { nullptr };
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
