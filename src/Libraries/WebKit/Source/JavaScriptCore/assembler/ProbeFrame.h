/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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

#if ENABLE(ASSEMBLER)

#include "CallFrame.h"
#include "ProbeStack.h"

namespace JSC {
namespace Probe {

class Frame {
public:
    Frame(void* frameBase, Stack& stack)
        : m_frameBase { static_cast<uint8_t*>(frameBase) }
        , m_stack { stack }
    { }

    template<typename T = JSValue>
    T argument(int argument)
    {
        return get<T>(CallFrame::argumentOffset(argument) * sizeof(Register));
    }
    template<typename T = JSValue>
    T operand(VirtualRegister operand)
    {
        return get<T>(operand.offset() * sizeof(Register));
    }
    template<typename T = JSValue>
    T operand(VirtualRegister operand, ptrdiff_t offset)
    {
        return get<T>(operand.offset() * sizeof(Register) + offset);
    }

    template<typename T>
    void setArgument(int argument, T value)
    {
        return set<T>(CallFrame::argumentOffset(argument) * sizeof(Register), value);
    }
    template<typename T>
    void setOperand(VirtualRegister operand, T value)
    {
        set<T>(operand.offset() * sizeof(Register), value);
    }
    template<typename T>
    void setOperand(VirtualRegister operand, ptrdiff_t offset, T value)
    {
        set<T>(operand.offset() * sizeof(Register) + offset, value);
    }

    template<typename T = JSValue>
    T get(ptrdiff_t offset)
    {
        return m_stack.get<T>(m_frameBase + offset);
    }
    template<typename T>
    void set(ptrdiff_t offset, T value)
    {
        m_stack.set<T>(m_frameBase + offset, value);
    }

private:
    uint8_t* m_frameBase;
    Stack& m_stack;
};

} // namespace Probe
} // namespace JSC

#endif // ENABLE(ASSEMBLER)
