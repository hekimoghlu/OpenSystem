/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#include <wtf/CheckedPtr.h>
#include "CachedResourceClient.h"
#include "ImageTypes.h"

namespace WebCore {

class CachedImage;
class Document;
class IntRect;

enum class VisibleInViewportState { Unknown, Yes, No };

class CachedImageClient : public CachedResourceClient, public CanMakeCheckedPtr<CachedImageClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CachedImageClient);
public:
    virtual ~CachedImageClient() = default;
    static CachedResourceClientType expectedType() { return ImageType; }
    CachedResourceClientType resourceClientType() const override { return expectedType(); }

    // Called whenever a frame of an image changes because we got more data from the network.
    // If not null, the IntRect is the changed rect of the image.
    virtual void imageChanged(CachedImage*, const IntRect* = nullptr) { }

    virtual bool canDestroyDecodedData() const { return true; }

    // Called when a new decoded frame for a large image is available or when an animated image is ready to advance to the next frame.
    virtual VisibleInViewportState imageFrameAvailable(CachedImage& image, ImageAnimatingState, const IntRect* changeRect) { imageChanged(&image, changeRect); return VisibleInViewportState::No; }
    virtual VisibleInViewportState imageVisibleInViewport(const Document&) const { return VisibleInViewportState::No; }

    virtual void didRemoveCachedImageClient(CachedImage&) { }

    virtual void scheduleRenderingUpdateForImage(CachedImage&) { }

    virtual bool allowsAnimation() const { return true; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE_CLIENT(CachedImageClient, ImageType);
