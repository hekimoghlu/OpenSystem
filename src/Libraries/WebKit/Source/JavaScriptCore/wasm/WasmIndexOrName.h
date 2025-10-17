/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#include "WasmName.h"
#include "WasmNameSection.h"
#include <wtf/RefPtr.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class LLIntOffsetsExtractor;

namespace Wasm {

struct NameSection;

// Keep this class copyable when the world is stopped: do not allocate any memory while copying this.
// SamplingProfiler copies it while suspending threads.
struct IndexOrName {
    friend class JSC::LLIntOffsetsExtractor;
    typedef size_t Index;

private:
#if USE(JSVALUE32_64)
    enum class Kind : uint8_t {
        Empty,
        Index,
        Name
    };
#endif
public:

    IndexOrName()
    {
#if USE(JSVALUE64)
        m_indexName.index = emptyTag;
#elif USE(JSVALUE32_64)
        m_kind = Kind::Empty;
#endif
    }

    IndexOrName(Index, std::pair<const Name*, RefPtr<NameSection>>&&);

    bool isEmpty() const
    {
#if USE(JSVALUE64)
        return std::bit_cast<Index>(m_indexName) & emptyTag;
#elif USE(JSVALUE32_64)
        return m_kind == Kind::Empty;
#endif
    }

    bool isIndex() const
    {
#if USE(JSVALUE64)
        return std::bit_cast<Index>(m_indexName) & indexTag;
#elif USE(JSVALUE32_64)
        return m_kind == Kind::Index;
#endif
    }

    bool isName() const
    {
        return !(isEmpty() || isIndex());
    }

    Index index() const
    {
        ASSERT(isIndex());
#if USE(JSVALUE64)
        return m_indexName.index & ~indexTag;
#elif USE(JSVALUE32_64)
        return m_indexName.index;
#endif
    }

    const Name* name() const
    {
        ASSERT(isName());
        return m_indexName.name;
    }

    NameSection* nameSection() const { return m_nameSection.get(); }
    void dump(PrintStream&) const;

private:
    union {
        Index index;
        const Name* name;
    } m_indexName;
    RefPtr<NameSection> m_nameSection;

#if USE(JSVALUE64)
    public:
    // Use the top bits as tags. Neither pointers nor the function index space should use them.
    static constexpr Index indexTag = 1ull << (CHAR_BIT * sizeof(Index) - 1);
    static constexpr Index emptyTag = 1ull << (CHAR_BIT * sizeof(Index) - 2);
    static constexpr Index allTags = indexTag | emptyTag;
    private:
#elif USE(JSVALUE32_64)
    // Use an explicit tag as pointers might have high bits set
    Kind m_kind;
#endif
};

String makeString(const IndexOrName&);

} } // namespace JSC::Wasm
