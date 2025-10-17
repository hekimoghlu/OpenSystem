/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "VirtualRegister.h"

#include "RegisterID.h"

namespace JSC {

void VirtualRegister::dump(PrintStream& out) const
{
    if (!isValid()) {
        out.print("<invalid>");
        return;
    }
    
    if (isHeader()) {
        if (m_virtualRegister == CallFrameSlot::codeBlock)
            out.print("codeBlock");
        else if (m_virtualRegister == CallFrameSlot::callee)
            out.print("callee");
        else if (m_virtualRegister == CallFrameSlot::argumentCountIncludingThis)
            out.print("argumentCountIncludingThis");
#if CPU(ADDRESS64)
        else if (!m_virtualRegister)
            out.print("callerFrame");
        else if (m_virtualRegister == 1)
            out.print("returnPC");
#else
        else if (!m_virtualRegister)
            out.print("callerFrameAndReturnPC");
#endif
        return;
    }
    
    if (isConstant()) {
        out.print("const", toConstantIndex());
        return;
    }
    
    if (isArgument()) {
        if (!toArgument())
            out.print("this");
        else
            out.print("arg", toArgument());
        return;
    }
    
    if (isLocal()) {
        out.print("loc", toLocal());
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}


VirtualRegister::VirtualRegister(RegisterID* reg)
    : VirtualRegister(reg->m_virtualRegister.m_virtualRegister)
{
}

VirtualRegister::VirtualRegister(RefPtr<RegisterID> reg)
    : VirtualRegister(reg.get())
{
}

} // namespace JSC
