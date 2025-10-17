/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

#include "TypeSet.h"

namespace JSC {

enum TypeProfilerGlobalIDFlags {
    TypeProfilerNeedsUniqueIDGeneration = -1,
    TypeProfilerNoGlobalIDExists = -2,
    TypeProfilerReturnStatement = -3
};

typedef intptr_t GlobalVariableID;

class TypeLocation {
public:
    TypeLocation()
        : m_instructionTypeSet(TypeSet::create())
        , m_globalTypeSet(nullptr)
        , m_divotForFunctionOffsetIfReturnStatement(UINT_MAX)
        , m_lastSeenType(TypeNothing)
    {
    }

    GlobalVariableID m_globalVariableID;
    RefPtr<TypeSet> m_instructionTypeSet;
    RefPtr<TypeSet> m_globalTypeSet;
    SourceID m_sourceID;
    unsigned m_divotStart;
    unsigned m_divotEnd;
    unsigned m_divotForFunctionOffsetIfReturnStatement;
    RuntimeType m_lastSeenType;
};

} // namespace JSC
