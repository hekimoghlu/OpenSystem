/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include "B3MathExtras.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlockInlines.h"
#include "B3CCallValue.h"
#include "B3Const32Value.h"
#include "B3ConstDoubleValue.h"
#include "B3ConstPtrValue.h"
#include "B3UpsilonValue.h"
#include "B3ValueInlines.h"
#include "JSCPtrTag.h"
#include "MathCommon.h"

namespace JSC { namespace B3 {

std::pair<BasicBlock*, Value*> powDoubleInt32(Procedure& procedure, BasicBlock* start, Origin origin, Value* x, Value* y)
{
    BasicBlock* functionCallCase = procedure.addBlock();
    BasicBlock* loopPreHeaderCase = procedure.addBlock();
    BasicBlock* loopTestForEvenCase = procedure.addBlock();
    BasicBlock* loopOdd = procedure.addBlock();
    BasicBlock* loopEvenOdd = procedure.addBlock();
    BasicBlock* continuation = procedure.addBlock();

    Value* shouldGoSlowPath = start->appendNew<Value>(procedure, Above, origin,
        y,
        start->appendNew<Const32Value>(procedure, origin, maxExponentForIntegerMathPow));
    start->appendNew<Value>(procedure, Branch, origin, shouldGoSlowPath);
    start->setSuccessors(FrequentedBlock(functionCallCase), FrequentedBlock(loopPreHeaderCase));

    // Function call.
    Value* yAsDouble = functionCallCase->appendNew<Value>(procedure, IToD, origin, y);
    Value* powResult = functionCallCase->appendNew<CCallValue>(
        procedure, Double, origin,
        functionCallCase->appendNew<ConstPtrValue>(procedure, origin, std::bit_cast<void*>(tagCFunction<OperationPtrTag>(Math::stdPowDouble))),
        x, yAsDouble);
    UpsilonValue* powResultUpsilon = functionCallCase->appendNew<UpsilonValue>(procedure, origin, powResult);
    functionCallCase->appendNew<Value>(procedure, Jump, origin);
    functionCallCase->setSuccessors(FrequentedBlock(continuation));

    // Loop pre-header.
    Value* initialResult = loopPreHeaderCase->appendNew<ConstDoubleValue>(procedure, origin, 1.);
    UpsilonValue* initialLoopValue = loopPreHeaderCase->appendNew<UpsilonValue>(procedure, origin, initialResult);
    UpsilonValue* initialResultValue = loopPreHeaderCase->appendNew<UpsilonValue>(procedure, origin, initialResult);
    UpsilonValue* initialSquaredInput = loopPreHeaderCase->appendNew<UpsilonValue>(procedure, origin, x);
    UpsilonValue* initialLoopCounter = loopPreHeaderCase->appendNew<UpsilonValue>(procedure, origin, y);
    loopPreHeaderCase->appendNew<Value>(procedure, Jump, origin);
    loopPreHeaderCase->setSuccessors(FrequentedBlock(loopTestForEvenCase));

    // Test if what is left of the counter is even.
    Value* inLoopCounter = loopTestForEvenCase->appendNew<Value>(procedure, Phi, Int32, origin);
    Value* inLoopSquaredInput = loopTestForEvenCase->appendNew<Value>(procedure, Phi, Double, origin);
    Value* lastCounterBit = loopTestForEvenCase->appendNew<Value>(procedure, BitAnd, origin,
        inLoopCounter,
        loopTestForEvenCase->appendNew<Const32Value>(procedure, origin, 1));
    loopTestForEvenCase->appendNew<Value>(procedure, Branch, origin, lastCounterBit);
    loopTestForEvenCase->setSuccessors(FrequentedBlock(loopOdd), FrequentedBlock(loopEvenOdd));

    // Counter is odd.
    Value* inLoopResult = loopOdd->appendNew<Value>(procedure, Phi, Double, origin);
    Value* updatedResult = loopOdd->appendNew<Value>(procedure, Mul, origin, inLoopResult, inLoopSquaredInput);
    UpsilonValue* updatedLoopResultUpsilon = loopOdd->appendNew<UpsilonValue>(procedure, origin, updatedResult);
    initialLoopValue->setPhi(inLoopResult);
    updatedLoopResultUpsilon->setPhi(inLoopResult);
    UpsilonValue* updatedLoopResult = loopOdd->appendNew<UpsilonValue>(procedure, origin, updatedResult);

    loopOdd->appendNew<Value>(procedure, Jump, origin);
    loopOdd->setSuccessors(FrequentedBlock(loopEvenOdd));

    // Even value and following the Odd.
    Value* squaredInput = loopEvenOdd->appendNew<Value>(procedure, Mul, origin, inLoopSquaredInput, inLoopSquaredInput);
    UpsilonValue* squaredInputUpsilon = loopEvenOdd->appendNew<UpsilonValue>(procedure, origin, squaredInput);
    initialSquaredInput->setPhi(inLoopSquaredInput);
    squaredInputUpsilon->setPhi(inLoopSquaredInput);

    Value* updatedCounter = loopEvenOdd->appendNew<Value>(procedure, ZShr, origin,
        inLoopCounter,
        loopEvenOdd->appendNew<Const32Value>(procedure, origin, 1));
    UpsilonValue* updatedCounterUpsilon = loopEvenOdd->appendNew<UpsilonValue>(procedure, origin, updatedCounter);
    initialLoopCounter->setPhi(inLoopCounter);
    updatedCounterUpsilon->setPhi(inLoopCounter);

    loopEvenOdd->appendNew<Value>(procedure, Branch, origin, updatedCounter);
    loopEvenOdd->setSuccessors(FrequentedBlock(loopTestForEvenCase), FrequentedBlock(continuation));

    // Inline loop.
    Value* finalResultPhi = continuation->appendNew<Value>(procedure, Phi, Double, origin);
    powResultUpsilon->setPhi(finalResultPhi);
    initialResultValue->setPhi(finalResultPhi);
    updatedLoopResult->setPhi(finalResultPhi);
    return std::make_pair(continuation, finalResultPhi);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

