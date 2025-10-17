/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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

#include "BytecodeConventions.h"
#include "CallFrame.h"
#include <wtf/PrintStream.h>

namespace JSC {

inline bool virtualRegisterIsLocal(int operand)
{
    return operand < 0;
}

inline bool virtualRegisterIsArgument(int operand)
{
    return operand >= 0;
}


class RegisterID;

class VirtualRegister {
public:
    friend VirtualRegister virtualRegisterForLocal(int);
    friend VirtualRegister virtualRegisterForArgumentIncludingThis(int, int);

    static constexpr int invalidVirtualRegister = 0x3fffffff;
    static constexpr int firstConstantRegisterIndex = FirstConstantRegisterIndex;

    VirtualRegister(RegisterID*);
    VirtualRegister(RefPtr<RegisterID>);

    VirtualRegister()
        : m_virtualRegister(invalidVirtualRegister)
    { }

    explicit VirtualRegister(int virtualRegister)
        : m_virtualRegister(virtualRegister)
    { }

    VirtualRegister(CallFrameSlot slot)
        : m_virtualRegister(static_cast<int>(slot))
    { }

    bool isValid() const { return (m_virtualRegister != invalidVirtualRegister); }
    bool isLocal() const { return virtualRegisterIsLocal(m_virtualRegister); }
    bool isArgument() const { return virtualRegisterIsArgument(m_virtualRegister); }
    bool isHeader() const { return m_virtualRegister >= 0 && m_virtualRegister < CallFrameSlot::thisArgument; }
    bool isConstant() const { return m_virtualRegister >= firstConstantRegisterIndex; }
    int toLocal() const { ASSERT(isLocal()); return operandToLocal(m_virtualRegister); }
    int toArgument() const { ASSERT(isArgument()); return operandToArgument(m_virtualRegister); }
    int toConstantIndex() const { ASSERT(isConstant()); return m_virtualRegister - firstConstantRegisterIndex; }
    int offset() const { return m_virtualRegister; }
    int offsetInBytes() const { return m_virtualRegister * sizeof(Register); }

    friend bool operator==(const VirtualRegister&, const VirtualRegister&) = default;
    bool operator<(VirtualRegister other) const { return m_virtualRegister < other.m_virtualRegister; }
    bool operator>(VirtualRegister other) const { return m_virtualRegister > other.m_virtualRegister; }
    bool operator<=(VirtualRegister other) const { return m_virtualRegister <= other.m_virtualRegister; }
    bool operator>=(VirtualRegister other) const { return m_virtualRegister >= other.m_virtualRegister; }

    VirtualRegister operator+(int value) const
    {
        return VirtualRegister(offset() + value);
    }
    VirtualRegister operator-(int value) const
    {
        return VirtualRegister(offset() - value);
    }
    VirtualRegister operator+(VirtualRegister value) const
    {
        return VirtualRegister(offset() + value.offset());
    }
    VirtualRegister operator-(VirtualRegister value) const
    {
        return VirtualRegister(offset() - value.offset());
    }
    VirtualRegister& operator+=(int value)
    {
        return *this = *this + value;
    }
    VirtualRegister& operator-=(int value)
    {
        return *this = *this - value;
    }
    
    void dump(PrintStream& out) const;

private:
    static int localToOperand(int local) { return -1 - local; }
    static int operandToLocal(int operand) { return -1 - operand; }
    static int operandToArgument(int operand) { return operand - CallFrame::thisArgumentOffset(); }
    static int argumentToOperand(int argument) { return argument + CallFrame::thisArgumentOffset(); }

    int m_virtualRegister;
};

static_assert(sizeof(VirtualRegister) == sizeof(int), "VirtualRegister is 32bit");

inline VirtualRegister virtualRegisterForLocal(int local)
{
    return VirtualRegister(VirtualRegister::localToOperand(local));
}

inline VirtualRegister virtualRegisterForArgumentIncludingThis(int argument, int offset = 0)
{
    return VirtualRegister(VirtualRegister::argumentToOperand(argument) + offset);
}

} // namespace JSC
