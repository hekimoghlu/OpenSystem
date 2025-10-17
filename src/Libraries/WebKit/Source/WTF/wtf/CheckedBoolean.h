/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#include <wtf/Assertions.h>

class CheckedBoolean {
    WTF_MAKE_FAST_ALLOCATED;
public:
#if ASSERT_ENABLED
    CheckedBoolean(const CheckedBoolean& other)
        : m_value(other.m_value)
        , m_checked(false)
    {
        other.m_checked = true;
    }
#endif

    CheckedBoolean(bool value)
        : m_value(value)
#if ASSERT_ENABLED
        , m_checked(false)
#endif
    {
    }
    
    ~CheckedBoolean()
    {
        ASSERT(m_checked);
    }
    
    operator bool()
    {
#if ASSERT_ENABLED
        m_checked = true;
#endif
        return m_value;
    }
    
private:
    bool m_value;
#if ASSERT_ENABLED
    mutable bool m_checked;
#endif
};
