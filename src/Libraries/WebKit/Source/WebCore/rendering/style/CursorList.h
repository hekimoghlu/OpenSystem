/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

#include "CursorData.h"
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class CursorList : public RefCounted<CursorList> {
public:
    static Ref<CursorList> create()
    {
        return adoptRef(*new CursorList);
    }

    const CursorData& operator[](int i) const { return m_vector[i]; }
    CursorData& operator[](int i) { return m_vector[i]; }
    const CursorData& at(size_t i) const { return m_vector.at(i); }
    CursorData& at(size_t i) { return m_vector.at(i); }

    bool operator==(const CursorList& o) const { return m_vector == o.m_vector; }

    size_t size() const { return m_vector.size(); }
    void append(const CursorData& cursorData) { m_vector.append(cursorData); }

private:
    CursorList()
    {
    }

    Vector<CursorData> m_vector;
};

} // namespace WebCore
