/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#pragma once

#include "SVGElement.h"
#include "SVGParserUtilities.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// Describe an SVG <hkern>/<vkern> element
struct SVGKerningPair {
    UnicodeRanges unicodeRange1;
    UncheckedKeyHashSet<String> unicodeName1;
    UncheckedKeyHashSet<String> glyphName1;

    UnicodeRanges unicodeRange2;
    UncheckedKeyHashSet<String> unicodeName2;
    UncheckedKeyHashSet<String> glyphName2;
    float kerning { 0 };
};

class SVGFontElement final : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFontElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFontElement);
public:
    static Ref<SVGFontElement> create(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFontElement, SVGElement>;

private:
    SVGFontElement(const QualifiedName&, Document&);

    bool rendererIsNeeded(const RenderStyle&) final { return false; }
};

} // namespace WebCore
