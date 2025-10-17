/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "MathMLRootElement.h"

#if ENABLE(MATHML)

#include "RenderMathMLRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLRootElement);

using namespace MathMLNames;

static RootType rootTypeOf(const QualifiedName& tagName)
{
    if (tagName.matches(msqrtTag))
        return RootType::SquareRoot;
    ASSERT(tagName.matches(mrootTag));
    return RootType::RootWithIndex;
}

inline MathMLRootElement::MathMLRootElement(const QualifiedName& tagName, Document& document)
    : MathMLRowElement(tagName, document)
    , m_rootType(rootTypeOf(tagName))
{
}

Ref<MathMLRootElement> MathMLRootElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLRootElement(tagName, document));
}

RenderPtr<RenderElement> MathMLRootElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(msqrtTag) || hasTagName(mrootTag));
    return createRenderer<RenderMathMLRoot>(*this, WTFMove(style));
}

}

#endif // ENABLE(MATHML)
