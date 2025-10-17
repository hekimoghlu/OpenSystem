/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

#include "StyleGeneratedImage.h"

namespace WebCore {

class CSSInvalidImageValue;

class StyleInvalidImage final : public StyleGeneratedImage {
public:
    static Ref<StyleInvalidImage> create();

    virtual ~StyleInvalidImage();

    bool operator==(const StyleImage&) const final { return false; }
    bool equals(const StyleInvalidImage&) const { return false; }
    bool canRender(const RenderElement*, float) const final { return false; }

    static constexpr bool isFixedSize = true;

protected:
    void didAddClient(RenderElement&) final { }
    void didRemoveClient(RenderElement&) final { }

    FloatSize fixedSize(const RenderElement&) const final { return { }; }

private:
    StyleInvalidImage();
    
    bool isPending() const final { return false; }
    void load(CachedResourceLoader&, const ResourceLoaderOptions&) final;
    bool knownToBeOpaque(const RenderElement&) const { return false; }

    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    Ref<CSSValue> computedStyleValue(const RenderStyle&) const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StyleInvalidImage, isInvalidImage)
