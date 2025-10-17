/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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

#include "DirectArguments.h"
#include "JSBigInt.h"
#include "JSLexicalEnvironment.h"
#include "JSModuleEnvironment.h"

namespace JSC {

inline constexpr bool isDynamicallySizedType(JSType type)
{
    if (type == DirectArgumentsType
        || type == FinalObjectType
        || type == LexicalEnvironmentType
        || type == ModuleEnvironmentType)
        return true;
    return false;
}

inline size_t cellSize(JSCell* cell)
{
    Structure* structure = cell->structure();
    const ClassInfo* classInfo = structure->classInfoForCells();
    JSType cellType = cell->type();

    if (isDynamicallySizedType(cellType)) {
        switch (cellType) {
        case DirectArgumentsType: {
            auto* args = jsCast<DirectArguments*>(cell);
            return DirectArguments::allocationSize(args->m_minCapacity);
        }
        case FinalObjectType:
            return JSFinalObject::allocationSize(structure->inlineCapacity());
        case LexicalEnvironmentType: {
            auto* env = jsCast<JSLexicalEnvironment*>(cell);
            return JSLexicalEnvironment::allocationSize(env->symbolTable());
        }
        case ModuleEnvironmentType: {
            auto* env = jsCast<JSModuleEnvironment*>(cell);
            return JSModuleEnvironment::allocationSize(env->symbolTable());
        }
        default:
            RELEASE_ASSERT_NOT_REACHED();
        }
    }
    return classInfo->staticClassSize;
}

} // namespace JSC
