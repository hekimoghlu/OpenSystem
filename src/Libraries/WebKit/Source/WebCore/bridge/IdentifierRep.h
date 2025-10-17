/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

#include <string.h>
#include <wtf/Assertions.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
    
class IdentifierRep {
    WTF_MAKE_TZONE_ALLOCATED(IdentifierRep);
public:
    WEBCORE_EXPORT static IdentifierRep* get(int);
    WEBCORE_EXPORT static IdentifierRep* get(const char*);

    WEBCORE_EXPORT static bool isValid(IdentifierRep*);
    
    bool isString() const { return m_isString; }

    int number() const { return m_isString ? 0 : m_value.m_number; }
    const char* string() const { return m_isString ? m_value.m_string : 0; }

private:
    explicit IdentifierRep(int number) 
        : m_isString(false)
    {
        m_value.m_number = number;
    }
    
    explicit IdentifierRep(const char* name)
        : m_isString(true)
    {
        m_value.m_string = fastStrDup(name);
    }
    
    ~IdentifierRep(); // Not implemented
    
    union {
        const char* m_string;
        int m_number;
    } m_value;
    bool m_isString;
};

} // namespace WebCore

