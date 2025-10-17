/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#include "RenderMultiColumnSpannerPlaceholder.h"

#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderMultiColumnFlow.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMultiColumnSpannerPlaceholder);

RenderPtr<RenderMultiColumnSpannerPlaceholder> RenderMultiColumnSpannerPlaceholder::createAnonymous(RenderMultiColumnFlow& fragmentedFlow, RenderBox& spanner, const RenderStyle& parentStyle)
{
    auto newStyle = RenderStyle::createAnonymousStyleWithDisplay(parentStyle, DisplayType::Block);
    newStyle.setClear(Clear::Both); // We don't want floats in the row preceding the spanner to continue on the other side.
    auto placeholder = createRenderer<RenderMultiColumnSpannerPlaceholder>(fragmentedFlow, spanner, WTFMove(newStyle));
    placeholder->initializeStyle();
    return placeholder;
}

RenderMultiColumnSpannerPlaceholder::RenderMultiColumnSpannerPlaceholder(RenderMultiColumnFlow& fragmentedFlow, RenderBox& spanner, RenderStyle&& style)
    : RenderBox(Type::MultiColumnSpannerPlaceholder, fragmentedFlow.document(), WTFMove(style), TypeFlag::IsBoxModelObject)
    , m_spanner(spanner)
    , m_fragmentedFlow(fragmentedFlow)
{
    ASSERT(isRenderMultiColumnSpannerPlaceholder());
}

RenderMultiColumnSpannerPlaceholder::~RenderMultiColumnSpannerPlaceholder() = default;

ASCIILiteral RenderMultiColumnSpannerPlaceholder::renderName() const
{
    return "RenderMultiColumnSpannerPlaceholder"_s;
}

} // namespace WebCore
