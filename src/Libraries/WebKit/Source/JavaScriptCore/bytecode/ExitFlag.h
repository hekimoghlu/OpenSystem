/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "ExitingInlineKind.h"

namespace WTF {
class PrintStream;
} // namespace WTF

namespace JSC {

class ExitFlag {
public:
    ExitFlag() { }
    
    ExitFlag(bool value, ExitingInlineKind inlineKind)
    {
        if (!value)
            return;
        
        switch (inlineKind) {
        case ExitFromAnyInlineKind:
            m_bits = trueNotInlined | trueInlined;
            break;
        case ExitFromNotInlined:
            m_bits = trueNotInlined;
            break;
        case ExitFromInlined:
            m_bits = trueInlined;
            break;
        }
    }
    
    ExitFlag operator|(const ExitFlag& other) const
    {
        ExitFlag result;
        result.m_bits = m_bits | other.m_bits;
        return result;
    }
    
    ExitFlag& operator|=(const ExitFlag& other)
    {
        *this = *this | other;
        return *this;
    }
    
    ExitFlag operator&(const ExitFlag& other) const
    {
        ExitFlag result;
        result.m_bits = m_bits & other.m_bits;
        return result;
    }
    
    ExitFlag& operator&=(const ExitFlag& other)
    {
        *this = *this & other;
        return *this;
    }
    
    explicit operator bool() const
    {
        return !!m_bits;
    }
    
    bool isSet(ExitingInlineKind inlineKind) const
    {
        return !!(*this & ExitFlag(true, inlineKind));
    }
    
    void dump(WTF::PrintStream&) const;
    
private:
    static constexpr uint8_t trueNotInlined = 1;
    static constexpr uint8_t trueInlined = 2;
    
    uint8_t m_bits { 0 };
};

} // namespace JSC

