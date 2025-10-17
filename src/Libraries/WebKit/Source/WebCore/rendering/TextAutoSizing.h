/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#if ENABLE(TEXT_AUTOSIZING)

#include "RenderStyle.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Document;
class Text;

// FIXME: We can probably get rid of this class entirely and use std::unique_ptr<RenderStyle> as key
// as long as we use the right hash traits.
class TextAutoSizingKey {
public:
    TextAutoSizingKey() = default;
    enum DeletedTag { Deleted };
    explicit TextAutoSizingKey(DeletedTag);
    TextAutoSizingKey(const RenderStyle&, unsigned hash);

    const RenderStyle* style() const { ASSERT(!isDeleted()); return m_style.get(); }
    bool isDeleted() const { return HashTraits<std::unique_ptr<RenderStyle>>::isDeletedValue(m_style); }

    unsigned hash() const { return m_hash; }

private:
    std::unique_ptr<RenderStyle> m_style;
    unsigned m_hash { 0 };
};

inline bool operator==(const TextAutoSizingKey& a, const TextAutoSizingKey& b)
{
    if (a.isDeleted() || b.isDeleted())
        return false;
    if (!a.style() || !b.style())
        return a.style() == b.style();
    return a.style()->equalForTextAutosizing(*b.style());
}

struct TextAutoSizingHash {
    static unsigned hash(const TextAutoSizingKey& key) { return key.hash(); }
    static bool equal(const TextAutoSizingKey& a, const TextAutoSizingKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

struct TextAutoSizingHashTranslator {
    static unsigned hash(const RenderStyle& style)
    {
        return style.hashForTextAutosizing();
    }

    static bool equal(const TextAutoSizingKey& key, const RenderStyle& style)
    {
        if (key.isDeleted() || !key.style())
            return false;
        return key.style()->equalForTextAutosizing(style);
    }

    static void translate(TextAutoSizingKey& key, const RenderStyle& style, unsigned hash)
    {
        key = { style, hash };
    }
};

class TextAutoSizingValue {
    WTF_MAKE_TZONE_ALLOCATED(TextAutoSizingValue);
public:
    TextAutoSizingValue() = default;
    ~TextAutoSizingValue();

    void addTextNode(Text&, float size);

    enum class StillHasNodes : bool { No, Yes };
    StillHasNodes adjustTextNodeSizes();

private:
    void reset();

    UncheckedKeyHashSet<RefPtr<Text>> m_autoSizedNodes;
};

struct TextAutoSizingTraits : HashTraits<TextAutoSizingKey> {
    static const bool emptyValueIsZero = true;
    static void constructDeletedValue(TextAutoSizingKey& slot) { new (NotNull, &slot) TextAutoSizingKey(TextAutoSizingKey::Deleted); }
    static bool isDeletedValue(const TextAutoSizingKey& value) { return value.isDeleted(); }
};

class TextAutoSizing {
    WTF_MAKE_TZONE_ALLOCATED(TextAutoSizing);
public:
    TextAutoSizing() = default;

    void addTextNode(Text&, float size);
    void updateRenderTree();
    void reset();

private:
    UncheckedKeyHashMap<TextAutoSizingKey, std::unique_ptr<TextAutoSizingValue>, TextAutoSizingHash, TextAutoSizingTraits> m_textNodes;
};

} // namespace WebCore

#endif // ENABLE(TEXT_AUTOSIZING)
