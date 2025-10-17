/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#include "SVGDefsElement.h"

#include "LegacyRenderSVGHiddenContainer.h"
#include "RenderSVGHiddenContainer.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGDefsElement);

inline SVGDefsElement::SVGDefsElement(const QualifiedName& tagName, Document& document)
    : SVGGraphicsElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::defsTag));
}

Ref<SVGDefsElement> SVGDefsElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGDefsElement(tagName, document));
}

bool SVGDefsElement::isValid() const
{
    return SVGTests::isValid();
}

RenderPtr<RenderElement> SVGDefsElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGHiddenContainer>(RenderObject::Type::SVGHiddenContainer, *this, WTFMove(style));
    return createRenderer<LegacyRenderSVGHiddenContainer>(RenderObject::Type::LegacySVGHiddenContainer, *this, WTFMove(style));
}

}
