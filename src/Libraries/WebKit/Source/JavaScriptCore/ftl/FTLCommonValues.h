/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

#if ENABLE(FTL_JIT)

#include "FTLAbbreviatedTypes.h"
#include "FTLValueRange.h"

namespace JSC {

namespace B3 {
class BasicBlock;
class Procedure;
}

namespace FTL {

class CommonValues {
public:
    CommonValues();

    void initializeConstants(B3::Procedure&, B3::BasicBlock*);
    
    LValue booleanTrue { nullptr };
    LValue booleanFalse { nullptr };
    LValue int32Zero { nullptr };
    LValue int32One { nullptr };
    LValue int64Zero { nullptr };
    LValue intPtrZero { nullptr };
    LValue intPtrOne { nullptr };
    LValue intPtrTwo { nullptr };
    LValue intPtrThree { nullptr };
    LValue intPtrEight { nullptr };
    LValue doubleZero { nullptr };
    LValue doubleEncodeOffsetAsDouble { nullptr };
#if USE(BIGINT32)
    LValue bigInt32Zero { nullptr };
#endif

    const unsigned rangeKind { 0 };
    const unsigned profKind { 0 };
    const LValue branchWeights { nullptr };
    
    const ValueRange nonNegativeInt32;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
