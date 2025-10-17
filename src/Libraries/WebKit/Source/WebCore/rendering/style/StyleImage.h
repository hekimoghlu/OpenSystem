/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

#include "CSSValue.h"
#include "FloatSize.h"
#include "Image.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/TypeCasts.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CachedImage;
class CachedResourceLoader;
class CSSValue;
class Document;
class RenderElement;
class RenderObject;
class RenderStyle;
struct ResourceLoaderOptions;

typedef const void* WrappedImagePtr;

class StyleImage : public RefCountedAndCanMakeWeakPtr<StyleImage> {
public:
    virtual ~StyleImage() = default;

    virtual bool operator==(const StyleImage& other) const = 0;

    // Computed Style representation.
    virtual Ref<CSSValue> computedStyleValue(const RenderStyle&) const = 0;

    // Opaque representation.
    virtual WrappedImagePtr data() const = 0;

    // Loading.
    virtual bool isPending() const = 0;
    virtual void load(CachedResourceLoader&, const ResourceLoaderOptions&) = 0;
    virtual bool isLoaded(const RenderElement*) const { return true; }
    virtual bool errorOccurred() const { return false; }
    virtual bool usesDataProtocol() const { return false; }
    virtual bool hasImage() const { return false; }
    virtual URL reresolvedURL(const Document&) const { return { }; }

    // Clients.
    virtual void addClient(RenderElement&) = 0;
    virtual void removeClient(RenderElement&) = 0;
    virtual bool hasClient(RenderElement&) const = 0;

    // Size / scale.
    virtual FloatSize imageSize(const RenderElement*, float multiplier) const = 0;
    virtual bool usesImageContainerSize() const = 0;
    virtual void computeIntrinsicDimensions(const RenderElement*, Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio) = 0;
    virtual bool imageHasRelativeWidth() const = 0;
    virtual bool imageHasRelativeHeight() const = 0;
    virtual float imageScaleFactor() const { return 1; }
    virtual bool imageHasNaturalDimensions() const { return true; }

    // Image.
    virtual RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine = false) const = 0;
    virtual StyleImage* selectedImage() { return this; }
    virtual const StyleImage* selectedImage() const { return this; }
    virtual CachedImage* cachedImage() const { return nullptr; }

    // Rendering.
    virtual bool canRender(const RenderElement*, float /*multiplier*/) const { return true; }
    virtual void setContainerContextForRenderer(const RenderElement&, const FloatSize&, float) = 0;
    virtual bool knownToBeOpaque(const RenderElement&) const = 0;

    // Derived type.
    ALWAYS_INLINE bool isCachedImage() const { return m_type == Type::CachedImage; }
    ALWAYS_INLINE bool isCursorImage() const { return m_type == Type::CursorImage; }
    ALWAYS_INLINE bool isImageSet() const { return m_type == Type::ImageSet; }
    ALWAYS_INLINE bool isGeneratedImage() const { return isFilterImage() || isCanvasImage() || isCrossfadeImage() || isGradientImage() || isNamedImage() || isPaintImage() || isInvalidImage(); }
    ALWAYS_INLINE bool isFilterImage() const { return m_type == Type::FilterImage; }
    ALWAYS_INLINE bool isCanvasImage() const { return m_type == Type::CanvasImage; }
    ALWAYS_INLINE bool isCrossfadeImage() const { return m_type == Type::CrossfadeImage; }
    ALWAYS_INLINE bool isGradientImage() const { return m_type == Type::GradientImage; }
    ALWAYS_INLINE bool isNamedImage() const { return m_type == Type::NamedImage; }
    ALWAYS_INLINE bool isPaintImage() const { return m_type == Type::PaintImage; }
    ALWAYS_INLINE bool isInvalidImage() const { return m_type == Type::InvalidImage; }

    bool hasCachedImage() const { return m_type == Type::CachedImage || selectedImage()->isCachedImage(); }

protected:
    enum class Type : uint8_t {
        CachedImage,
        CursorImage,
        ImageSet,
        FilterImage,
        CanvasImage,
        CrossfadeImage,
        GradientImage,
        NamedImage,
        InvalidImage,
        PaintImage,
    };

    StyleImage(Type type)
        : m_type { type }
    {
    }

    Type m_type;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(ToClassName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::StyleImage& image) { return image.predicate(); } \
SPECIALIZE_TYPE_TRAITS_END()
