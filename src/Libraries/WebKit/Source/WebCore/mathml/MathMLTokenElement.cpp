/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include "MathMLTokenElement.h"

#if ENABLE(MATHML)

#include "HTTPParsers.h"
#include "MathMLNames.h"
#include "RenderMathMLToken.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLTokenElement);

using namespace MathMLNames;

MathMLTokenElement::MathMLTokenElement(const QualifiedName& tagName, Document& document)
    : MathMLPresentationElement(tagName, document, TypeFlag::HasCustomStyleResolveCallbacks)
{
}

Ref<MathMLTokenElement> MathMLTokenElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLTokenElement(tagName, document));
}

void MathMLTokenElement::didAttachRenderers()
{
    MathMLPresentationElement::didAttachRenderers();
    if (CheckedPtr mathmlRenderer = dynamicDowncast<RenderMathMLToken>(renderer()))
        mathmlRenderer->updateTokenContent();
}

void MathMLTokenElement::childrenChanged(const ChildChange& change)
{
    MathMLPresentationElement::childrenChanged(change);
    if (CheckedPtr mathmlRenderer = dynamicDowncast<RenderMathMLToken>(renderer()))
        mathmlRenderer->updateTokenContent();
}

RenderPtr<RenderElement> MathMLTokenElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(MathMLNames::miTag) || hasTagName(MathMLNames::mnTag) || hasTagName(MathMLNames::msTag) || hasTagName(MathMLNames::mtextTag));

    return createRenderer<RenderMathMLToken>(RenderObject::Type::MathMLToken, *this, WTFMove(style));
}

bool MathMLTokenElement::childShouldCreateRenderer(const Node& child) const
{
    // The HTML specification defines <mi>, <mo>, <mn>, <ms> and <mtext> as insertion points.
    return StyledElement::childShouldCreateRenderer(child);
}

std::optional<char32_t> MathMLTokenElement::convertToSingleCodePoint(StringView string)
{
    auto codePoints = string.trim(isASCIIWhitespaceWithoutFF<UChar>).codePoints();
    auto iterator = codePoints.begin();
    if (iterator == codePoints.end())
        return std::nullopt;
    std::optional<char32_t> character = *iterator;
    ++iterator;
    return iterator == codePoints.end() ? character : std::nullopt;
}

}

#endif // ENABLE(MATHML)
