/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#include "CSSParserContext.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class StyleSheetContents;

namespace Style {

class StyleSheetContentsCache {
public:
    static StyleSheetContentsCache& singleton();

    using Key = std::pair<String, CSSParserContext>;

    RefPtr<StyleSheetContents> get(const Key&);
    void add(Key&&, Ref<StyleSheetContents>);

    void clear();

private:
    friend class NeverDestroyed<StyleSheetContentsCache>;

    StyleSheetContentsCache();

    UncheckedKeyHashMap<Key, Ref<StyleSheetContents>> m_cache;
};

}
}
