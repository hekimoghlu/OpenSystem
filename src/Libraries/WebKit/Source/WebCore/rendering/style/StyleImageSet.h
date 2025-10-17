/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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

#include "StyleMultiImage.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class StyleImageSet final : public StyleMultiImage {
    WTF_MAKE_TZONE_ALLOCATED(StyleImageSet);
public:
    static Ref<StyleImageSet> create(Vector<ImageWithScale>&&, Vector<size_t>&&);
    virtual ~StyleImageSet();

    bool operator==(const StyleImage& other) const;
    bool equals(const StyleImageSet&) const;

    ImageWithScale selectBestFitImage(const Document&) final;

private:
    explicit StyleImageSet(Vector<ImageWithScale>&&, Vector<size_t>&&);

    Ref<CSSValue> computedStyleValue(const RenderStyle&) const final;

    ImageWithScale bestImageForScaleFactor();
    void updateDeviceScaleFactor(const Document&);

    bool m_accessedBestFitImage { false };
    ImageWithScale m_bestFitImage;
    float m_deviceScaleFactor { 1 };
    Vector<ImageWithScale> m_images;
    Vector<size_t> m_sortedIndices;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StyleImageSet, isImageSet)
