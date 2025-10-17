/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "ProgressShadowElement.h"

#include "HTMLNames.h"
#include "HTMLProgressElement.h"
#include "RenderProgress.h"
#include "RenderStyleInlines.h"
#include "UserAgentParts.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ProgressShadowElement);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ProgressInnerElement);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ProgressBarElement);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ProgressValueElement);

using namespace HTMLNames;

ProgressShadowElement::ProgressShadowElement(Document& document)
    : HTMLDivElement(HTMLNames::divTag, document)
{
}

HTMLProgressElement* ProgressShadowElement::progressElement() const
{
    return downcast<HTMLProgressElement>(shadowHost());
}

bool ProgressShadowElement::rendererIsNeeded(const RenderStyle& style)
{
    RenderObject* progressRenderer = progressElement()->renderer();
    return progressRenderer && !progressRenderer->style().hasUsedAppearance() && HTMLDivElement::rendererIsNeeded(style);
}

ProgressInnerElement::ProgressInnerElement(Document& document)
    : ProgressShadowElement(document)
{
}

RenderPtr<RenderElement> ProgressInnerElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderProgress>(*this, WTFMove(style));
}

bool ProgressInnerElement::rendererIsNeeded(const RenderStyle& style)
{
    auto* progressRenderer = progressElement()->renderer();
    return progressRenderer && !progressRenderer->style().hasUsedAppearance() && HTMLDivElement::rendererIsNeeded(style);
}

ProgressBarElement::ProgressBarElement(Document& document)
    : ProgressShadowElement(document)
{
}

ProgressValueElement::ProgressValueElement(Document& document)
    : ProgressShadowElement(document)
{
}

void ProgressValueElement::setInlineSizePercentage(double size)
{
    setInlineStyleProperty(CSSPropertyInlineSize, size, CSSUnitType::CSS_PERCENTAGE);
}

Ref<ProgressInnerElement> ProgressInnerElement::create(Document& document)
{
    Ref<ProgressInnerElement> result = adoptRef(*new ProgressInnerElement(document));
    result->setUserAgentPart(UserAgentParts::webkitProgressInnerElement());
    return result;
}

Ref<ProgressBarElement> ProgressBarElement::create(Document& document)
{
    Ref<ProgressBarElement> result = adoptRef(*new ProgressBarElement(document));
    result->setUserAgentPart(UserAgentParts::webkitProgressBar());
    return result;
}

Ref<ProgressValueElement> ProgressValueElement::create(Document& document)
{
    Ref<ProgressValueElement> result = adoptRef(*new ProgressValueElement(document));
    result->setUserAgentPart(UserAgentParts::webkitProgressValue());
    return result;
}

}
