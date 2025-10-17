/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#include "SVGFontFaceSrcElement.h"

#include "CSSFontFaceSrcValue.h"
#include "CSSValueList.h"
#include "ElementChildIteratorInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFontFaceElement.h"
#include "SVGFontFaceNameElement.h"
#include "SVGFontFaceUriElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFontFaceSrcElement);

using namespace SVGNames;
    
inline SVGFontFaceSrcElement::SVGFontFaceSrcElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(font_face_srcTag));
}

Ref<SVGFontFaceSrcElement> SVGFontFaceSrcElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFontFaceSrcElement(tagName, document));
}

Ref<CSSValueList> SVGFontFaceSrcElement::createSrcValue() const
{
    CSSValueListBuilder list;
    for (auto& child : childrenOfType<SVGElement>(*this)) {
        if (RefPtr element = dynamicDowncast<SVGFontFaceUriElement>(child)) {
            if (auto srcValue = element->createSrcValue(); !srcValue->isEmpty())
                list.append(WTFMove(srcValue));
        } else if (RefPtr element = dynamicDowncast<SVGFontFaceNameElement>(child)) {
            if (auto srcValue = element->createSrcValue(); !srcValue->isEmpty())
                list.append(WTFMove(srcValue));
        }
    }
    return CSSValueList::createCommaSeparated(WTFMove(list));
}

void SVGFontFaceSrcElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);
    if (RefPtr parent = dynamicDowncast<SVGFontFaceElement>(parentNode()))
        parent->rebuildFontFace();
}

}
