/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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

#include <unicode/uchar.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

enum BidiEmbeddingSource { FromStyleOrDOM, FromUnicode };

// Used to keep track of explicit embeddings.
class BidiContext : public ThreadSafeRefCounted<BidiContext> {
public:
    WEBCORE_EXPORT static Ref<BidiContext> create(unsigned char level, UCharDirection, bool override = false, BidiEmbeddingSource = FromStyleOrDOM, BidiContext* parent = nullptr);

    BidiContext* parent() const { return m_parent.get(); }
    unsigned char level() const { return m_level; }
    UCharDirection dir() const { return static_cast<UCharDirection>(m_direction); }
    bool override() const { return m_override; }
    BidiEmbeddingSource source() const { return static_cast<BidiEmbeddingSource>(m_source); }

    WEBCORE_EXPORT Ref<BidiContext> copyStackRemovingUnicodeEmbeddingContexts();

private:
    BidiContext(unsigned char level, UCharDirection, bool override, BidiEmbeddingSource, BidiContext* parent);

    static Ref<BidiContext> createUncached(unsigned char level, UCharDirection, bool override, BidiEmbeddingSource, BidiContext* parent);

    unsigned m_level : 6; // The maximium bidi level is 62: http://unicode.org/reports/tr9/#Explicit_Levels_and_Directions
    unsigned m_direction : 5; // Direction
    unsigned m_override : 1;
    unsigned m_source : 1; // BidiEmbeddingSource
    RefPtr<BidiContext> m_parent;
};

inline unsigned char nextGreaterOddLevel(unsigned char level)
{
    return (level + 1) | 1;
}

inline unsigned char nextGreaterEvenLevel(unsigned char level)
{
    return (level + 2) & ~1;
}

bool operator==(const BidiContext&, const BidiContext&);

} // namespace WebCore
