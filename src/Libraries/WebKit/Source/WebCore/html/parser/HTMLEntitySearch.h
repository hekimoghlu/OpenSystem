/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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

#include <unicode/umachine.h>

namespace WebCore {

struct HTMLEntityTableEntry;

class HTMLEntitySearch {
public:
    HTMLEntitySearch();

    void advance(UChar);

    bool isEntityPrefix() const { return m_first; }
    unsigned currentLength() const { return m_currentLength; }

    const HTMLEntityTableEntry* match() const { return m_mostRecentMatch; }

private:
    enum CompareResult { Before, Prefix, After };
    CompareResult compare(const HTMLEntityTableEntry*, UChar) const;
    const HTMLEntityTableEntry* findFirst(UChar) const;
    const HTMLEntityTableEntry* findLast(UChar) const;

    void fail()
    {
        m_first = nullptr;
        m_last = nullptr;
    }

    unsigned m_currentLength { 0 };
    const HTMLEntityTableEntry* m_mostRecentMatch { nullptr };
    const HTMLEntityTableEntry* m_first { nullptr };
    const HTMLEntityTableEntry* m_last { nullptr };
};

} // namespace WebCore
