/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "CSSStyleSheet.h"
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;

namespace ContentExtensions {

class ContentExtensionStyleSheet : public RefCounted<ContentExtensionStyleSheet> {
public:
    static Ref<ContentExtensionStyleSheet> create(Document& document)
    {
        return adoptRef(*new ContentExtensionStyleSheet(document));
    }
    virtual ~ContentExtensionStyleSheet();

    bool addDisplayNoneSelector(const String& selector, uint32_t selectorID);

    CSSStyleSheet& styleSheet() { return m_styleSheet.get(); }

private:
    ContentExtensionStyleSheet(Document&);

    Ref<CSSStyleSheet> m_styleSheet;
    UncheckedKeyHashSet<uint32_t, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_addedSelectorIDs;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
