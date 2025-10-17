/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
#include "YouTubeEmbedShadowElement.h"

#include "RenderBlockFlow.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(YouTubeEmbedShadowElement);

Ref<YouTubeEmbedShadowElement> YouTubeEmbedShadowElement::create(Document& document)
{
    auto element = adoptRef(*new YouTubeEmbedShadowElement(document));
    element->setInlineStyleProperty(CSSPropertyAll, CSSValueInitial);
    return element;
}

YouTubeEmbedShadowElement::YouTubeEmbedShadowElement(Document& document)
    : HTMLDivElement(HTMLNames::divTag, document)
{
}

RenderPtr<RenderElement> YouTubeEmbedShadowElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderBlockFlow>(RenderObject::Type::BlockFlow, *this, WTFMove(style));
}

}
