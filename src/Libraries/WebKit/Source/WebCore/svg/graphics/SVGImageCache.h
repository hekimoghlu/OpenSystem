/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "Image.h"
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CachedImage;
class CachedImageClient;
class ImageBuffer;
class LayoutSize;
class SVGImage;
class SVGImageForContainer;
class RenderObject;

class SVGImageCache {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SVGImageCache, WEBCORE_EXPORT);
public:
    explicit SVGImageCache(SVGImage*);
    ~SVGImageCache();

    void removeClientFromCache(const CachedImageClient*);

    void setContainerContextForClient(const CachedImageClient&, const LayoutSize&, float, const URL&);
    FloatSize imageSizeForRenderer(const RenderObject*) const;

    Image* imageForRenderer(const RenderObject*) const;

private:
    Image* findImageForRenderer(const RenderObject*) const;
    RefPtr<SVGImage> protectedSVGImage() const;

    typedef UncheckedKeyHashMap<const CachedImageClient*, RefPtr<SVGImageForContainer>> ImageForContainerMap;

    WeakPtr<SVGImage> m_svgImage;
    ImageForContainerMap m_imageForContainerMap;
};

} // namespace WebCore
