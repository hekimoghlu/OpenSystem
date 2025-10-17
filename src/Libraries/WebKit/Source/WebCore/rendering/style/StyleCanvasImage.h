/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

#include "CanvasObserver.h"
#include "HTMLCanvasElement.h"
#include "StyleGeneratedImage.h"
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;

class StyleCanvasImage final : public StyleGeneratedImage, public CanvasObserver {
public:
    static Ref<StyleCanvasImage> create(String name)
    {
        return adoptRef(*new StyleCanvasImage(WTFMove(name)));
    }
    virtual ~StyleCanvasImage();

    bool operator==(const StyleImage&) const final;
    bool equals(const StyleCanvasImage&) const;
    
    static constexpr bool isFixedSize = true;

private:
    explicit StyleCanvasImage(String&&);

    Ref<CSSValue> computedStyleValue(const RenderStyle&) const final;
    bool isPending() const final;
    void load(CachedResourceLoader&, const ResourceLoaderOptions&) final;
    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    bool knownToBeOpaque(const RenderElement&) const final;
    FloatSize fixedSize(const RenderElement&) const final;
    void didAddClient(RenderElement&) final;
    void didRemoveClient(RenderElement&) final;

    // CanvasObserver.
    bool isStyleCanvasImage() const final { return true; }
    void canvasChanged(CanvasBase&, const FloatRect&) final;
    void canvasResized(CanvasBase&) final;
    void canvasDestroyed(CanvasBase&) final;

    HTMLCanvasElement* element(Document&) const;

    // The name of the canvas.
    String m_name;
    // The document supplies the element and owns it.
    mutable WeakPtr<HTMLCanvasElement, WeakPtrImplWithEventTargetData> m_element;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyleCanvasImage)
    static bool isType(const WebCore::StyleImage& image) { return image.isCanvasImage(); }
    static bool isType(const WebCore::CanvasObserver& canvasObserver) { return canvasObserver.isStyleCanvasImage(); }
SPECIALIZE_TYPE_TRAITS_END()

