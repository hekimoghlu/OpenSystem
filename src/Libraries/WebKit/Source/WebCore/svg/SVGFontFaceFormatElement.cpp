/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "SVGFontFaceFormatElement.h"

#include "SVGElementTypeHelpers.h"
#include "SVGFontFaceElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFontFaceFormatElement);
    
using namespace SVGNames;
    
inline SVGFontFaceFormatElement::SVGFontFaceFormatElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(font_face_formatTag));
}

Ref<SVGFontFaceFormatElement> SVGFontFaceFormatElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFontFaceFormatElement(tagName, document));
}

void SVGFontFaceFormatElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);

    if (!parentNode() || !parentNode()->hasTagName(font_face_uriTag))
        return;
    
    RefPtr ancestor = parentNode()->parentNode();
    if (!ancestor || !ancestor->hasTagName(font_face_srcTag))
        return;
    
    ancestor = ancestor->parentNode();
    if (RefPtr fontFaceElement = dynamicDowncast<SVGFontFaceElement>(ancestor))
        fontFaceElement->rebuildFontFace();
}

}
