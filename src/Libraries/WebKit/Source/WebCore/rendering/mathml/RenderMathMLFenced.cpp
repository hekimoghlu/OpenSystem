/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#include "RenderMathMLFenced.h"

#if ENABLE(MATHML)

#include "ElementInlines.h"
#include "FontSelector.h"
#include "MathMLNames.h"
#include "MathMLRowElement.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderInline.h"
#include "RenderMathMLFencedOperator.h"
#include "RenderText.h"
#include "RenderTreeBuilder.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

using namespace MathMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLFenced);

static constexpr auto gOpeningBraceChar = "("_s;
static constexpr auto gClosingBraceChar = ")"_s;

RenderMathMLFenced::RenderMathMLFenced(MathMLRowElement& element, RenderStyle&& style)
    : RenderMathMLRow(Type::MathMLFenced, element, WTFMove(style))
{
    ASSERT(isRenderMathMLFenced());
}

RenderMathMLFenced::~RenderMathMLFenced() = default;

void RenderMathMLFenced::updateFromElement()
{
    const Ref fenced = element();

    // The open operator defaults to a left parenthesis.
    auto& open = fenced->attributeWithoutSynchronization(MathMLNames::openAttr);
    m_open = open.isNull() ? gOpeningBraceChar : open;

    // The close operator defaults to a right parenthesis.
    auto& close = fenced->attributeWithoutSynchronization(MathMLNames::closeAttr);
    m_close = close.isNull() ? gClosingBraceChar : close;

    auto& separators = fenced->attributeWithoutSynchronization(MathMLNames::separatorsAttr);
    if (!separators.isNull()) {
        StringBuilder characters;
        for (unsigned i = 0; i < separators.length(); i++) {
            if (!deprecatedIsSpaceOrNewline(separators[i]))
                characters.append(separators[i]);
        }
        m_separators = !characters.length() ? 0 : characters.toString().impl();
    } else {
        // The separator defaults to a single comma.
        m_separators = StringImpl::create(","_s);
    }

    if (firstChild()) {
        // FIXME: The mfenced element fails to update dynamically when its open, close and separators attributes are changed (https://bugs.webkit.org/show_bug.cgi?id=57696).
        if (auto* fencedOperator = dynamicDowncast<RenderMathMLFencedOperator>(*firstChild()))
            fencedOperator->updateOperatorContent(m_open);
        m_closeFenceRenderer->updateOperatorContent(m_close);
    }
}

}

#endif
