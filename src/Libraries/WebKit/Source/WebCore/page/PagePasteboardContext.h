/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

#include "PageIdentifier.h"
#include "PasteboardContext.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

class PagePasteboardContext final : public PasteboardContext {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PagePasteboardContext);
public:
    PagePasteboardContext(std::optional<PageIdentifier>&& pageID = std::nullopt)
        : PasteboardContext()
        , m_pageID(WTFMove(pageID))
    {
    }

    ~PagePasteboardContext() { }

    static std::unique_ptr<PasteboardContext> create(std::optional<PageIdentifier>&& pageID = std::nullopt)
    {
        return makeUnique<PagePasteboardContext>(WTFMove(pageID));
    }

    std::optional<PageIdentifier> pageID() const { return m_pageID; }
    bool isPagePasteboardContext() const override { return true; }

private:
    std::optional<PageIdentifier> m_pageID;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PagePasteboardContext)
    static bool isType(const WebCore::PasteboardContext& context) { return context.isPagePasteboardContext(); }
SPECIALIZE_TYPE_TRAITS_END()

