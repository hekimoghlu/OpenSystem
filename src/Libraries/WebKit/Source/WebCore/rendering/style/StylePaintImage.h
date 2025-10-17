/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSVariableData;

class StylePaintImage final : public StyleGeneratedImage {
public:
    static Ref<StylePaintImage> create(String name, Ref<CSSVariableData> arguments)
    {
        return adoptRef(*new StylePaintImage(WTFMove(name), WTFMove(arguments)));
    }
    virtual ~StylePaintImage();

    static constexpr bool isFixedSize = false;

private:
    explicit StylePaintImage(String&&, Ref<CSSVariableData>&&);

    bool operator==(const StyleImage&) const final;

    Ref<CSSValue> computedStyleValue(const RenderStyle&) const final;
    bool isPending() const final;
    void load(CachedResourceLoader&, const ResourceLoaderOptions&) final;
    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    bool knownToBeOpaque(const RenderElement&) const final;
    FloatSize fixedSize(const RenderElement&) const final;
    void didAddClient(RenderElement&) final { }
    void didRemoveClient(RenderElement&) final { }

    String m_name;
    Ref<CSSVariableData> m_arguments;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StylePaintImage, isPaintImage)
