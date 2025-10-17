/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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

#include "FloatSize.h"
#include "ImageOrientation.h"
#include "IntSize.h"
#include "Path.h"
#include "TextFlags.h"
#include "TextIndicator.h"
#include <wtf/Forward.h>

#if PLATFORM(IOS_FAMILY)
#include <wtf/RetainPtr.h>
typedef struct CGImage *CGImageRef;
#elif PLATFORM(MAC)
#include <wtf/RetainPtr.h>
OBJC_CLASS NSImage;
#elif PLATFORM(WIN)
typedef struct HBITMAP__* HBITMAP;
#elif USE(CAIRO)
#include "RefPtrCairo.h"
#elif USE(SKIA)
#include <skia/core/SkImage.h>
#endif

namespace WebCore {

class Element;
class Image;
class IntRect;
class LocalFrame;
class Node;

#if PLATFORM(IOS_FAMILY)
typedef RetainPtr<CGImageRef> DragImageRef;
#elif PLATFORM(MAC)
typedef RetainPtr<NSImage> DragImageRef;
#elif PLATFORM(WIN)
typedef HBITMAP DragImageRef;
#elif USE(CAIRO)
typedef RefPtr<cairo_surface_t> DragImageRef;
#elif USE(SKIA)
typedef sk_sp<SkImage> DragImageRef;
#else
typedef void* DragImageRef;
#endif

#if PLATFORM(COCOA)
extern const float ColorSwatchCornerRadius;
extern const float ColorSwatchStrokeSize;
extern const float ColorSwatchWidth;
#endif

IntSize dragImageSize(DragImageRef);

// These functions should be memory neutral, eg. if they return a newly allocated image,
// they should release the input image. As a corollary these methods don't guarantee
// the input image ref will still be valid after they have been called.
DragImageRef fitDragImageToMaxSize(DragImageRef, const IntSize& srcSize, const IntSize& dstSize);
DragImageRef scaleDragImage(DragImageRef, FloatSize scale);
DragImageRef platformAdjustDragImageForDeviceScaleFactor(DragImageRef, float deviceScaleFactor);
DragImageRef dissolveDragImageToFraction(DragImageRef, float delta);

DragImageRef createDragImageFromImage(Image*, ImageOrientation);
DragImageRef createDragImageIconForCachedImageFilename(const String&);

WEBCORE_EXPORT DragImageRef createDragImageForNode(LocalFrame&, Node&);
WEBCORE_EXPORT DragImageRef createDragImageForSelection(LocalFrame&, TextIndicatorData&, bool forceBlackText = false);
WEBCORE_EXPORT DragImageRef createDragImageForRange(LocalFrame&, const SimpleRange&, bool forceBlackText = false);
DragImageRef createDragImageForColor(const Color&, const FloatRect&, float, Path&);
DragImageRef createDragImageForImage(LocalFrame&, Node&, IntRect& imageRect, IntRect& elementRect);
DragImageRef createDragImageForLink(Element&, URL&, const String& label, TextIndicatorData&, float deviceScaleFactor);
void deleteDragImage(DragImageRef);

IntPoint dragOffsetForLinkDragImage(DragImageRef);
FloatPoint anchorPointForLinkDragImage(DragImageRef);

class DragImage final {
public:
    WEBCORE_EXPORT DragImage();
    explicit DragImage(DragImageRef);
    WEBCORE_EXPORT DragImage(DragImage&&);
    WEBCORE_EXPORT ~DragImage();

    DragImage(std::optional<TextIndicatorData>&& indicatorData, std::optional<Path>&& visiblePath)
        : m_indicatorData(WTFMove(indicatorData))
        , m_visiblePath(WTFMove(visiblePath))
    { }

    WEBCORE_EXPORT DragImage& operator=(DragImage&&);

    void setIndicatorData(const TextIndicatorData& data) { m_indicatorData = data; }
    bool hasIndicatorData() const { return !!m_indicatorData; }
    const std::optional<TextIndicatorData>& indicatorData() const { return m_indicatorData; }

    void setVisiblePath(const Path& path) { m_visiblePath = path; }
    bool hasVisiblePath() const { return !!m_visiblePath; }
    const std::optional<Path>& visiblePath() const { return m_visiblePath; }

    explicit operator bool() const { return !!m_dragImageRef; }
    DragImageRef get() const { return m_dragImageRef; }

private:
    DragImageRef m_dragImageRef;
    std::optional<TextIndicatorData> m_indicatorData;
    std::optional<Path> m_visiblePath;
};

}
