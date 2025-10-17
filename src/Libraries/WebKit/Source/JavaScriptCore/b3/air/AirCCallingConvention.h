/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#include "AirArg.h"
#include "AirInst.h"
#include "B3Type.h"
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class CCallValue;
class BasicBlock;

namespace Air {

class Code;

Vector<Arg> computeCCallingConvention(Code&, CCallValue*);

size_t cCallResultCount(Code&, CCallValue*);
bool cCallArgumentEvenRegisterAlignment(Type);

/*
 * On some platforms (well, on 32-bit platforms,) C functions can take arguments
 * that need more than one Air::Arg to pass around. These functions serve as a
 * source of truth about how args of a CCallValue must be represented by the time we
 * lower to Air.
 */

// Return the number of Air::Args needed to marshall this Value to the C function
size_t cCallArgumentRegisterCount(Type);
// Return the width of the individual Air::Args needed to marshall this value
Width cCallArgumentRegisterWidth(Type);

Tmp cCallResult(Code&, CCallValue*, unsigned);

Inst buildCCall(Code&, Value* origin, const Vector<Arg>&);

template<unsigned size> constexpr inline Type b3IntegerType;
template<> constexpr inline Type b3IntegerType<4> = Int32;
template<> constexpr inline Type b3IntegerType<8> = Int64;

template<typename T> constexpr inline Type b3Type;
template<> constexpr inline Type b3Type<int> = b3IntegerType<sizeof(int)>;
template<> constexpr inline Type b3Type<long> = b3IntegerType<sizeof(long)>;
template<> constexpr inline Type b3Type<long long> = b3IntegerType<sizeof(long long)>;
template<> constexpr inline Type b3Type<unsigned> = b3IntegerType<sizeof(unsigned)>;
template<> constexpr inline Type b3Type<unsigned long> = b3IntegerType<sizeof(unsigned long)>;
template<> constexpr inline Type b3Type<unsigned long long> = b3IntegerType<sizeof(unsigned long long)>;
template<> constexpr inline Type b3Type<float> = Float;
template<> constexpr inline Type b3Type<double> = Double;
template<typename T> constexpr inline Type b3Type<T*> = pointerType();

// This maps between B3 args and air args.
// On 32-bit platforms only, this may be an n:m mapping.
struct ArgumentValueList {
#if CPU(ARM_THUMB2)
    JS_EXPORT_PRIVATE Value* makeStitch(B3::BasicBlock*, Value* hi, Value* low) const;
#endif
    JS_EXPORT_PRIVATE Value* makeCCallValue(B3::BasicBlock*, Type, Air::Arg) const;
    JS_EXPORT_PRIVATE Value* makeCCallValue(B3::BasicBlock*, size_t idx) const;

    inline Value* operator[](size_t idx) const
    {
        return makeCCallValue(block, idx);
    }

    inline Value* withBlock(B3::BasicBlock *bb, size_t idx) const
    {
        return makeCCallValue(bb, idx);
    }

    inline Vector<Value*> eager() const
    {
        Vector<Value*> result;
        for (size_t i = 0; i < types.size(); ++i)
            result.append(makeCCallValue(block, i));
        return result;
    }

    Procedure& procedure;
    B3::BasicBlock* block;

    Vector<Type> types;
    Vector<Air::Arg> underlyingArgs;
    Vector<unsigned> argUnderlyingCounts;
};

JS_EXPORT_PRIVATE ArgumentValueList computeCCallArguments(Procedure&, B3::BasicBlock*, const Vector<Type>&);

template<typename ... T>
ArgumentValueList cCallArgumentValues(Procedure& procedure, B3::BasicBlock* block)
{
    return computeCCallArguments(procedure, block, { b3Type<T> ... });
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
