/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "ImagePaintingOptions.h"
#include "IntSize.h"
#include "PlatformImage.h"
#include "RenderingResource.h"
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

#if USE(SKIA)
class GLFence;
#endif

class GraphicsContext;
class NativeImageBackend;

class NativeImage final : public RenderingResource {
    WTF_MAKE_TZONE_ALLOCATED(NativeImage);
public:
    static WEBCORE_EXPORT RefPtr<NativeImage> create(PlatformImagePtr&&, RenderingResourceIdentifier = RenderingResourceIdentifier::generate());
    // Creates a NativeImage that is intended to be drawn once or only few times. Signals the platform to avoid generating any caches for the image.
    static WEBCORE_EXPORT RefPtr<NativeImage> createTransient(PlatformImagePtr&&, RenderingResourceIdentifier = RenderingResourceIdentifier::generate());

    virtual ~NativeImage();

    WEBCORE_EXPORT const PlatformImagePtr& platformImage() const;

    WEBCORE_EXPORT IntSize size() const;
    bool hasAlpha() const;
    std::optional<Color> singlePixelSolidColor() const;
    WEBCORE_EXPORT DestinationColorSpace colorSpace() const;
    WEBCORE_EXPORT Headroom headroom() const;

    void draw(GraphicsContext&, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions);
    void clearSubimages();

    WEBCORE_EXPORT void replaceBackend(UniqueRef<NativeImageBackend>);
    NativeImageBackend& backend() { return m_backend.get(); }
    const NativeImageBackend& backend() const { return m_backend.get(); }

#if USE(COORDINATED_GRAPHICS)
    uint64_t uniqueID() const;
#endif
protected:
    NativeImage(UniqueRef<NativeImageBackend>, RenderingResourceIdentifier);

    bool isNativeImage() const final { return true; }

    UniqueRef<NativeImageBackend> m_backend;
};

class NativeImageBackend {
public:
    WEBCORE_EXPORT NativeImageBackend();
    WEBCORE_EXPORT virtual ~NativeImageBackend();
    virtual const PlatformImagePtr& platformImage() const = 0;
    virtual IntSize size() const = 0;
    virtual bool hasAlpha() const = 0;
    virtual DestinationColorSpace colorSpace() const = 0;
    virtual Headroom headroom() const = 0;
    WEBCORE_EXPORT virtual bool isRemoteNativeImageBackendProxy() const;

#if USE(SKIA)
    // During DisplayList recording a fence is created, so that we can wait until the SkImage finished rendering
    // before we attempt to access the GPU resource from a secondary thread during replay (in threaded GPU painting mode).
    virtual void finishAcceleratedRenderingAndCreateFence() { }
    virtual void waitForAcceleratedRenderingFenceCompletion() { }

    virtual const GrDirectContext* skiaGrContext() const { return nullptr; }

    // Use to copy an accelerated NativeImage, cloning the PlatformImageNativeImageBackend, creating
    // a new SkImage tied to the current thread (and thus the thread-local GrDirectContext), but re-using
    // the existing backend texture, of this NativeImage. This avoids any GPU->GPU copies and has the
    // sole purpose to abe able to access an accelerated NativeImage from another thread, that is not
    // the creation thread.
    virtual RefPtr<NativeImage> copyAcceleratedNativeImageBorrowingBackendTexture() const { return nullptr; }
#endif
};

class PlatformImageNativeImageBackend final : public NativeImageBackend {
public:
    WEBCORE_EXPORT PlatformImageNativeImageBackend(PlatformImagePtr);
    WEBCORE_EXPORT ~PlatformImageNativeImageBackend() final;
    WEBCORE_EXPORT const PlatformImagePtr& platformImage() const final;
    WEBCORE_EXPORT IntSize size() const final;
    WEBCORE_EXPORT bool hasAlpha() const final;
    WEBCORE_EXPORT DestinationColorSpace colorSpace() const final;
    WEBCORE_EXPORT Headroom headroom() const final;

#if USE(SKIA)
    void finishAcceleratedRenderingAndCreateFence() final;
    void waitForAcceleratedRenderingFenceCompletion() final;

    const GrDirectContext* skiaGrContext() const final;

    RefPtr<NativeImage> copyAcceleratedNativeImageBorrowingBackendTexture() const final;
#endif
private:
    PlatformImagePtr m_platformImage;
#if USE(SKIA)
    std::unique_ptr<GLFence> m_fence WTF_GUARDED_BY_LOCK(m_fenceLock);
    Lock m_fenceLock;
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::NativeImage)
    static bool isType(const WebCore::RenderingResource& renderingResource) { return renderingResource.isNativeImage(); }
SPECIALIZE_TYPE_TRAITS_END()
