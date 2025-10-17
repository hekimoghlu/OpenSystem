/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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

#include "Document.h"
#include "SecurityOriginData.h"
#include "TextBreakingPositionContext.h"
#include "Timer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {
namespace Layout {

class TextBreakingPositionCache {
    WTF_MAKE_TZONE_ALLOCATED(TextBreakingPositionCache);
public:
    static constexpr size_t minimumRequiredTextLengthForContentBreakCache = 5;
    static constexpr size_t minimumRequiredContentBreaks = 3;

    static TextBreakingPositionCache& singleton();

    TextBreakingPositionCache();

    using Key = std::tuple<String, TextBreakingPositionContext, SecurityOriginData>;
    using List = Vector<size_t, 8>;
    void set(const Key&, List&& breakingPositionList);
    const List* get(const Key&) const;

    void clear();

private:
    void evict();

private:
    using TextBreakingPositionMap = UncheckedKeyHashMap<Key, List>;
    TextBreakingPositionMap m_breakingPositionMap;
    size_t m_cachedContentSize { 0 };
    Timer m_delayedEvictionTimer;
};

} // namespace Layout
} // namespace WebCore
