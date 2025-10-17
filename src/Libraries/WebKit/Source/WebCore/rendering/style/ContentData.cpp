/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
#include "config.h"
#include "ContentData.h"

#include "RenderCounter.h"
#include "RenderImage.h"
#include "RenderImageResource.h"
#include "RenderImageResourceStyleImage.h"
#include "RenderQuote.h"
#include "RenderStyle.h"
#include "RenderTextFragment.h"
#include "StyleInheritedData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ContentData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageContentData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(TextContentData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CounterContentData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(QuoteContentData);

std::unique_ptr<ContentData> ContentData::clone() const
{
    auto result = cloneInternal();
    auto* lastNewData = result.get();
    for (auto* contentData = next(); contentData; contentData = contentData->next()) {
        lastNewData->setNext(contentData->cloneInternal());
        lastNewData = lastNewData->next();
    }
    return result;
}

RenderPtr<RenderObject> ImageContentData::createContentRenderer(Document& document, const RenderStyle& pseudoStyle) const
{
    auto image = createRenderer<RenderImage>(RenderObject::Type::Image, document, RenderStyle::createStyleInheritingFromPseudoStyle(pseudoStyle), const_cast<StyleImage*>(m_image.ptr()));
    image->initializeStyle();
    image->setAltText(altText());
    return image;
}

RenderPtr<RenderObject> TextContentData::createContentRenderer(Document& document, const RenderStyle&) const
{
    auto fragment = createRenderer<RenderTextFragment>(document, m_text);
    fragment->setAltText(altText());
    return fragment;
}

RenderPtr<RenderObject> CounterContentData::createContentRenderer(Document& document, const RenderStyle&) const
{
    return createRenderer<RenderCounter>(document, *m_counter);
}

RenderPtr<RenderObject> QuoteContentData::createContentRenderer(Document& document, const RenderStyle& pseudoStyle) const
{
    auto quote = createRenderer<RenderQuote>(document, RenderStyle::createStyleInheritingFromPseudoStyle(pseudoStyle), m_quote);
    quote->initializeStyle();
    return quote;
}

} // namespace WebCore
