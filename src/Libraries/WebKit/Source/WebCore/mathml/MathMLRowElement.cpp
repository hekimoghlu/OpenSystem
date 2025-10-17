/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#include "MathMLRowElement.h"

#if ENABLE(MATHML)

#include "MathMLNames.h"
#include "MathMLOperatorElement.h"
#include "RenderMathMLFenced.h"
#include "RenderMathMLMenclose.h"
#include "RenderMathMLRow.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLRowElement);

using namespace MathMLNames;

MathMLRowElement::MathMLRowElement(const QualifiedName& tagName, Document& document, OptionSet<TypeFlag> constructionType)
    : MathMLPresentationElement(tagName, document, constructionType)
{
}

Ref<MathMLRowElement> MathMLRowElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLRowElement(tagName, document));
}

void MathMLRowElement::childrenChanged(const ChildChange& change)
{
    // FIXME: Avoid this invalidation for valid MathMLFractionElement/MathMLScriptsElement.
    // See https://bugs.webkit.org/show_bug.cgi?id=276828.
    for (RefPtr child = firstChild(); child; child = child->nextSibling()) {
        if (child->hasTagName(moTag))
            static_cast<MathMLOperatorElement*>(child.get())->setOperatorFormDirty();
    }

    MathMLPresentationElement::childrenChanged(change);
}

RenderPtr<RenderElement> MathMLRowElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (hasTagName(mfencedTag))
        return createRenderer<RenderMathMLFenced>(*this, WTFMove(style));

    ASSERT(hasTagName(merrorTag) || hasTagName(mphantomTag) || hasTagName(mrowTag) || hasTagName(mstyleTag));
    return createRenderer<RenderMathMLRow>(RenderObject::Type::MathMLRow, *this, WTFMove(style));
}

bool MathMLRowElement::acceptsMathVariantAttribute()
{
    return hasTagName(mstyleTag);
}

}

#endif // ENABLE(MATHML)
