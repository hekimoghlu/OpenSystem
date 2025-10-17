/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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

#include "B3Const32Value.h"
#include "B3Const64Value.h"

namespace JSC { namespace B3 {

// Usually you want to use Const32Value or Const64Value directly. But this is useful for writing
// platform-agnostic code. Note that a ConstPtrValue will behave like either a Const32Value or
// Const64Value depending on platform.

#if CPU(ADDRESS64)
typedef Const64Value ConstPtrValueBase;
#else
typedef Const32Value ConstPtrValueBase;
#endif

class ConstPtrValue : public ConstPtrValueBase {
public:
    void* value() const
    {
        return std::bit_cast<void*>(ConstPtrValueBase::value());
    }

private:
    friend class Procedure;
    friend class Value;

    template<typename T>
    static Opcode opcodeFromConstructor(Origin, T*) { return ConstPtrValueBase::opcodeFromConstructor(); }
    template<typename T>
    ConstPtrValue(Origin origin, T* pointer)
        : ConstPtrValueBase(origin, std::bit_cast<intptr_t>(pointer))
    {
    }
    template<typename T>
    static Opcode opcodeFromConstructor(Origin, T) { return ConstPtrValueBase::opcodeFromConstructor(); }
    template<typename T>
    ConstPtrValue(Origin origin, T pointer)
        : ConstPtrValueBase(origin, static_cast<intptr_t>(pointer))
    {
    }
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
