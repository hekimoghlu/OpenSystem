/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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

#include "Color.h"
#include "DecodingOptions.h"
#include "FloatRect.h"
#include "FloatSize.h"
#include "GraphicsTypes.h"
#include "ImageAdapter.h"
#include "ImageOrientation.h"
#include "ImagePaintingOptions.h"
#include "ImageTypes.h"
#include "NativeImage.h"
#include "Timer.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class AffineTransform;
class FloatPoint;
class FloatSize;
class GraphicsContext;
class FragmentedSharedBuffer;
class ShareableBitmap;
struct Length;

// This class gets notified when an image creates or destroys decoded frames and when it advances animation frames.
class ImageObserver;

class Image : public RefCountedAndCanMakeWeakPtr<Image> {
    friend class CachedSubimage;
    friend class GraphicsContext;
public:
    virtual ~Image();
    
    WEBCORE_EXPORT static RefPtr<Image> create(ImageObserver&);
    WEBCORE_EXPORT static std::optional<Ref<Image>> create(RefPtr<ShareableBitmap>&&);
    WEBCORE_EXPORT static bool supportsType(const String&);

    virtual bool isBitmapImage() const { return false; }
    virtual bool isGeneratedImage() const { return false; }
    virtual bool isCrossfadeGeneratedImage() const { return false; }
    virtual bool isNamedImageGeneratedImage() const { return false; }
    virtual bool isGradientImage() const { return false; }
    virtual bool isSVGImage() const { return false; }
    virtual bool isSVGImageForContainer() const { return false; }
    virtual bool isSVGResourceImage() const { return false; }
    virtual bool isPDFDocumentImage() const { return false; }
    virtual bool isCustomPaintImage() const { return false; }

    bool drawsSVGImage() const { return isSVGImage() || isSVGImageForContainer(); }

    virtual unsigned frameCount() const { return 1; }

    virtual bool currentFrameKnownToBeOpaque() const = 0;
    virtual bool isAnimated() const { return false; }

    // Derived classes should override this if their rendering could leak
    // cross-origin data (outside of the resource itself, which undergoes
    // a CORS cross-origin check).
    virtual bool renderingTaintsOrigin() const { return false; }

    WEBCORE_EXPORT static Image& nullImage();
    bool isNull() const { return size().isEmpty(); }

    virtual void setContainerSize(const FloatSize&) { }
    virtual bool usesContainerSize() const { return false; }
    virtual bool hasRelativeWidth() const { return false; }
    virtual bool hasRelativeHeight() const { return false; }
    virtual void computeIntrinsicDimensions(Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio);

    virtual FloatSize size(ImageOrientation = ImageOrientation::Orientation::FromImage) const = 0;
    virtual FloatSize sourceSize(ImageOrientation orientation = ImageOrientation::Orientation::FromImage) const { return size(orientation); }
    virtual bool hasDensityCorrectedSize() const { return false; }
    FloatRect rect() const { return FloatRect(FloatPoint(), size()); }
    float width() const { return size().width(); }
    float height() const { return size().height(); }
    virtual std::optional<IntPoint> hotSpot() const { return std::nullopt; }
    virtual ImageOrientation orientation() const { return ImageOrientation::Orientation::FromImage; }

    WEBCORE_EXPORT EncodedDataStatus setData(RefPtr<FragmentedSharedBuffer>&& data, bool allDataReceived);
    virtual EncodedDataStatus dataChanged(bool /*allDataReceived*/) { return EncodedDataStatus::Unknown; }

    virtual String uti() const { return String(); } // null string if unknown
    virtual String filenameExtension() const { return String(); } // null string if unknown
    virtual String accessibilityDescription() const { return String(); } // null string if unknown

    virtual void destroyDecodedData(bool /*destroyAll*/ = true) { }

    FragmentedSharedBuffer* data() { return m_encodedImageData.get(); }
    const FragmentedSharedBuffer* data() const { return m_encodedImageData.get(); }

    virtual DestinationColorSpace colorSpace();

    // Animation begins whenever someone draws the image, so startAnimation() is not normally called.
    // It will automatically pause once all observers no longer want to render the image anywhere.
    virtual void startAnimation() { }
    void startAnimationAsynchronously();
    virtual void stopAnimation() {}
    virtual void resetAnimation() {}
    virtual bool isAnimating() const { return false; }
    bool animationPending() const { return m_animationStartTimer && m_animationStartTimer->isActive(); }
    std::optional<bool> allowsAnimation() const { return m_allowsAnimation; }
    void setAllowsAnimation(std::optional<bool> allowsAnimation) { m_allowsAnimation = allowsAnimation; }
    static bool systemAllowsAnimationControls() { return gSystemAllowsAnimationControls; }
    WEBCORE_EXPORT static void setSystemAllowsAnimationControls(bool allowsControls);

    // Typically the CachedImage that owns us.
    RefPtr<ImageObserver> imageObserver() const;
    void setImageObserver(RefPtr<ImageObserver>&&);

    WEBCORE_EXPORT ImageAdapter& adapter();
    void invalidateAdapter();

    URL sourceURL() const;
    WEBCORE_EXPORT String mimeType() const;
    long long expectedContentLength() const;

    enum TileRule { StretchTile, RoundTile, SpaceTile, RepeatTile };

    virtual RefPtr<NativeImage> nativeImage(const DestinationColorSpace& = DestinationColorSpace::SRGB()) { return nullptr; }
    virtual RefPtr<NativeImage> nativeImageAtIndex(unsigned) { return nativeImage(); }
    virtual RefPtr<NativeImage> currentNativeImage() { return nativeImage(); }
    virtual RefPtr<NativeImage> currentPreTransformedNativeImage(ImageOrientation = ImageOrientation::Orientation::FromImage) { return currentNativeImage(); }

    virtual void drawPattern(GraphicsContext&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { });

#if ASSERT_ENABLED
    virtual bool hasSolidColor() { return false; }
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
    virtual bool shouldUseQuickLookForFullscreen() const { return false; }
#endif

#if ENABLE(SPATIAL_IMAGE_DETECTION)
    virtual bool isSpatial() const { return false; }
#endif

    virtual void dump(WTF::TextStream&) const;

    WEBCORE_EXPORT RefPtr<ShareableBitmap> toShareableBitmap() const;

protected:
    WEBCORE_EXPORT Image(ImageObserver* = nullptr);

    static void fillWithSolidColor(GraphicsContext&, const FloatRect& dstRect, const Color&, CompositeOperator);

    virtual bool shouldDrawFromCachedSubimage(GraphicsContext&) const { return false; }
    virtual bool mustDrawFromCachedSubimage(GraphicsContext&) const { return false; }
    virtual ImageDrawResult draw(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { }) = 0;
    ImageDrawResult drawTiled(GraphicsContext&, const FloatRect& dstRect, const FloatPoint& srcPoint, const FloatSize& tileSize, const FloatSize& spacing, ImagePaintingOptions = { });
    ImageDrawResult drawTiled(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, const FloatSize& tileScaleFactor, TileRule hRule, TileRule vRule, ImagePaintingOptions = { });

    // Supporting tiled drawing
    virtual std::optional<Color> singlePixelSolidColor() const { return std::nullopt; }

private:
    RefPtr<FragmentedSharedBuffer> m_encodedImageData;
    WeakPtr<ImageObserver> m_imageObserver;
    std::unique_ptr<ImageAdapter> m_adapter;

    // A value of true or false will override the default Page::imageAnimationEnabled state.
    std::optional<bool> m_allowsAnimation { std::nullopt };
    std::unique_ptr<Timer> m_animationStartTimer;
    static bool gSystemAllowsAnimationControls;
};

WTF::TextStream& operator<<(WTF::TextStream&, const Image&);

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_IMAGE(ToClassName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::Image& image) { return image.is##ToClassName(); } \
SPECIALIZE_TYPE_TRAITS_END()
