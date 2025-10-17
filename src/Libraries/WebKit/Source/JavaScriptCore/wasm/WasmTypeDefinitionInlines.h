/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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

#if ENABLE(WEBASSEMBLY)

#include "WasmTypeDefinition.h"

namespace JSC { namespace Wasm {

inline TypeInformation& TypeInformation::singleton()
{
    static TypeInformation* theOne;
    static std::once_flag typeInformationFlag;

    std::call_once(typeInformationFlag, [] () {
        theOne = new TypeInformation;
    });
    return *theOne;
}

inline TypeIndex TypeDefinition::index() const
{
    ASSERT(refCount() > 1); // TypeInformation::m_typeSet + caller
    return std::bit_cast<TypeIndex>(this);
}

inline const TypeDefinition& TypeInformation::get(TypeIndex index)
{
    ASSERT(index != TypeDefinition::invalidIndex);
    auto def = std::bit_cast<const TypeDefinition*>(index);
    ASSERT(def->refCount() > 1); // TypeInformation::m_typeSet + caller
    return *def;
}

inline const FunctionSignature& TypeInformation::getFunctionSignature(TypeIndex index)
{
    const TypeDefinition& signature = get(index).expand();
    ASSERT(signature.is<FunctionSignature>());
    return *signature.as<FunctionSignature>();
}

inline std::optional<const FunctionSignature*> TypeInformation::tryGetFunctionSignature(TypeIndex index)
{
    const TypeDefinition& signature = get(index).expand();
    if (signature.is<FunctionSignature>())
        return signature.as<FunctionSignature>();
    return std::nullopt;
}

// TODO: merge with TypeDefinition::index().
inline TypeIndex TypeInformation::get(const TypeDefinition& type)
{
    if (ASSERT_ENABLED) {
        TypeInformation& info = singleton();
        Locker locker { info.m_lock };
        ASSERT_UNUSED(info, info.m_typeSet.contains(TypeHash { const_cast<TypeDefinition&>(type) }));
    }
    return type.index();
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
