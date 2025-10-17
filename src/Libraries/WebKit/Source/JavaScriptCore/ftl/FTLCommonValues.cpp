/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "FTLCommonValues.h"

#include "B3BasicBlockInlines.h"
#include "B3Const32Value.h"
#include "B3Const64Value.h"
#include "B3ConstDoubleValue.h"
#include "B3ConstPtrValue.h"
#include "B3ValueInlines.h"

#if ENABLE(FTL_JIT)

namespace JSC { namespace FTL {

using namespace B3;

CommonValues::CommonValues()
{
}

void CommonValues::initializeConstants(B3::Procedure& proc, B3::BasicBlock* block)
{
    int32Zero = block->appendNew<Const32Value>(proc, Origin(), 0);
    int32One = block->appendNew<Const32Value>(proc, Origin(), 1);
    booleanTrue = int32One;
    booleanFalse = int32Zero;
    int64Zero = block->appendNew<Const64Value>(proc, Origin(), 0);
    intPtrZero = block->appendNew<ConstPtrValue>(proc, Origin(), 0);
    intPtrOne = block->appendNew<ConstPtrValue>(proc, Origin(), 1);
    intPtrTwo = block->appendNew<ConstPtrValue>(proc, Origin(), 2);
    intPtrThree = block->appendNew<ConstPtrValue>(proc, Origin(), 3);
    intPtrEight = block->appendNew<ConstPtrValue>(proc, Origin(), 8);
    doubleZero = block->appendNew<ConstDoubleValue>(proc, Origin(), 0.);
    doubleEncodeOffsetAsDouble = block->appendNew<ConstDoubleValue>(proc, Origin(), std::bit_cast<double>(JSValue::DoubleEncodeOffset));
#if USE(BIGINT32)
    bigInt32Zero = block->appendNew<Const64Value>(proc, Origin(), JSValue::BigInt32Tag);
#endif
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

