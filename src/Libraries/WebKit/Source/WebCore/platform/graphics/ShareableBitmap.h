/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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

#include "CopyImageOptions.h"
#include "DestinationColorSpace.h"
#include "IntRect.h"
#include "PlatformImage.h"
#include "SharedMemory.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/DebugHeap.h>
#include <wtf/ExportMacros.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>

#if USE(SKIA)
#include <skia/core/SkImageInfo.h>
#endif

namespace WebCore {

class GraphicsContext;
class Image;
class NativeImage;

class ShareableBitmapConfiguration {
public:
    ShareableBitmapConfiguration() = default;

    WEBCORE_EXPORT ShareableBitmapConfiguration(const IntSize&, std::optional<DestinationColorSpace> = std::nullopt, bool isOpaque = false);
    WEBCORE_EXPORT ShareableBitmapConfiguration(const IntSize&, std::optional<DestinationColorSpace>, bool isOpaque, unsigned bytesPerPixel, unsigned bytesPerRow
#if USE(CG)
        , CGBitmapInfo
#endif
    );
#if USE(CG)
    ShareableBitmapConfiguration(NativeImage&);
#endif

    IntSize size() const { return m_size; }
    const DestinationColorSpace& colorSpace() const { return m_colorSpace ? *m_colorSpace : DestinationColorSpace::SRGB(); }
    PlatformColorSpaceValue platformColorSpace() const { return colorSpace().platformColorSpace(); }
    bool isOpaque() const { return m_isOpaque; }

    unsigned bytesPerPixel() const { ASSERT(!m_bytesPerPixel.hasOverflowed()); return m_bytesPerPixel; }
    unsigned bytesPerRow() const { ASSERT(!m_bytesPerRow.hasOverflowed()); return m_bytesPerRow; }
#if USE(CG)
    CGBitmapInfo bitmapInfo() const { return m_bitmapInfo; }
#endif
#if USE(SKIA)
    const SkImageInfo& imageInfo() const { return m_imageInfo; }
#endif

    CheckedUint32 sizeInBytes() const { return m_bytesPerRow * m_size.height(); }

    WEBCORE_EXPORT static CheckedUint32 calculateBytesPerRow(const IntSize&, const DestinationColorSpace&);
    WEBCORE_EXPORT static CheckedUint32 calculateSizeInBytes(const IntSize&, const DestinationColorSpace&);

private:
    friend struct IPC::ArgumentCoder<ShareableBitmapConfiguration, void>;

    static std::optional<DestinationColorSpace> validateColorSpace(std::optional<DestinationColorSpace>);
    static CheckedUint32 calculateBytesPerPixel(const DestinationColorSpace&);
#if USE(CG)
    static CGBitmapInfo calculateBitmapInfo(const DestinationColorSpace&, bool isOpaque);
#endif

    IntSize m_size;
    std::optional<DestinationColorSpace> m_colorSpace;
    bool m_isOpaque { false };

    CheckedUint32 m_bytesPerPixel;
    CheckedUint32 m_bytesPerRow;
#if USE(CG)
    CGBitmapInfo m_bitmapInfo { 0 };
#endif
#if USE(SKIA)
    SkImageInfo m_imageInfo;
#endif
};

class ShareableBitmapHandle  {
public:
    ShareableBitmapHandle(ShareableBitmapHandle&&) = default;
    explicit ShareableBitmapHandle(const ShareableBitmapHandle&) = default;
    WEBCORE_EXPORT ShareableBitmapHandle(SharedMemory::Handle&&, const ShareableBitmapConfiguration&);

    ShareableBitmapHandle& operator=(ShareableBitmapHandle&&) = default;

    SharedMemory::Handle& handle() { return m_handle; }

    // Take ownership of the memory for process memory accounting purposes.
    WEBCORE_EXPORT void takeOwnershipOfMemory(MemoryLedger) const;
    // Transfer ownership of the memory for process memory accounting purposes.
    WEBCORE_EXPORT void setOwnershipOfMemory(const ProcessIdentity&, MemoryLedger) const;

private:
    friend struct IPC::ArgumentCoder<ShareableBitmapHandle, void>;
    friend class ShareableBitmap;

    SharedMemory::Handle m_handle;
    ShareableBitmapConfiguration m_configuration;
};

class ShareableBitmap : public ThreadSafeRefCounted<ShareableBitmap> {
public:
    using Handle = ShareableBitmapHandle;

    // Create a shareable bitmap whose backing memory can be shared with another process.
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> create(const ShareableBitmapConfiguration&);

    // Create a shareable bitmap from an already existing shared memory block.
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> create(const ShareableBitmapConfiguration&, Ref<SharedMemory>&&);

    // Create a shareable bitmap from a NativeImage.
#if USE(CG)
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> createFromImagePixels(NativeImage&);
#endif
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> createFromImageDraw(NativeImage&, const DestinationColorSpace&);
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> createFromImageDraw(NativeImage&, const DestinationColorSpace&, const IntSize&);
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> createFromImageDraw(NativeImage&, const DestinationColorSpace&, const IntSize& destinationSize, const IntSize& sourceSize);

    // Create a shareable bitmap from a handle.
    WEBCORE_EXPORT static RefPtr<ShareableBitmap> create(Handle&&, SharedMemory::Protection = SharedMemory::Protection::ReadWrite);

    // Create a shareable bitmap from a ReadOnly handle.
    WEBCORE_EXPORT static std::optional<Ref<ShareableBitmap>> createReadOnly(std::optional<Handle>&&);

    WEBCORE_EXPORT std::optional<Handle> createHandle(SharedMemory::Protection = SharedMemory::Protection::ReadWrite) const;

    // Create a ReadOnly handle.
    WEBCORE_EXPORT std::optional<Handle> createReadOnlyHandle() const;

    WEBCORE_EXPORT void setOwnershipOfMemory(const ProcessIdentity&);

    IntSize size() const { return m_configuration.size(); }
    IntRect bounds() const { return IntRect(IntPoint(), size()); }

    WEBCORE_EXPORT std::span<const uint8_t> span() const;
    WEBCORE_EXPORT std::span<uint8_t> mutableSpan();
    size_t bytesPerRow() const { return m_configuration.bytesPerRow(); }
    size_t sizeInBytes() const { return m_configuration.sizeInBytes(); }

    // Create a graphics context that can be used to paint into the backing store.
    WEBCORE_EXPORT std::unique_ptr<GraphicsContext> createGraphicsContext();

    // Paint the backing store into the given context.
    WEBCORE_EXPORT void paint(GraphicsContext&, const IntPoint& destination, const IntRect& source);
    WEBCORE_EXPORT void paint(GraphicsContext&, float scaleFactor, const IntPoint& destination, const IntRect& source);

    // This creates a bitmap image that directly references the shared bitmap data.
    // This is only safe to use when we know that the contents of the shareable bitmap won't change.
    WEBCORE_EXPORT RefPtr<Image> createImage();

#if USE(CG)
    // This creates a copied CGImageRef (most likely a copy-on-write) of the shareable bitmap.
    WEBCORE_EXPORT RetainPtr<CGImageRef> makeCGImageCopy();

    // This creates a CGImageRef that directly references the shared bitmap data.
    // This is only safe to use when we know that the contents of the shareable bitmap won't change.
    WEBCORE_EXPORT RetainPtr<CGImageRef> makeCGImage(ShouldInterpolate = ShouldInterpolate::No);

    WEBCORE_EXPORT PlatformImagePtr createPlatformImage(BackingStoreCopy = CopyBackingStore, ShouldInterpolate = ShouldInterpolate::No);
#elif USE(CAIRO)
    // This creates a BitmapImage that directly references the shared bitmap data.
    // This is only safe to use when we know that the contents of the shareable bitmap won't change.
    WEBCORE_EXPORT RefPtr<cairo_surface_t> createPersistentCairoSurface();
    WEBCORE_EXPORT RefPtr<cairo_surface_t> createCairoSurface();

    PlatformImagePtr createPlatformImage(BackingStoreCopy = CopyBackingStore, ShouldInterpolate = ShouldInterpolate::No) { return createCairoSurface(); }
#elif USE(SKIA)
    WEBCORE_EXPORT PlatformImagePtr createPlatformImage(BackingStoreCopy = CopyBackingStore, ShouldInterpolate = ShouldInterpolate::No);
#endif

private:
    ShareableBitmap(ShareableBitmapConfiguration, Ref<SharedMemory>&&);

#if USE(CG)
    RetainPtr<CGImageRef> createCGImage(CGDataProviderRef, ShouldInterpolate) const;
    static void releaseBitmapContextData(void* typelessBitmap, void* typelessData);
#endif

#if USE(CAIRO)
    static void releaseSurfaceData(void* typelessBitmap);
#endif

    ShareableBitmapConfiguration m_configuration;
    Ref<SharedMemory> m_sharedMemory;
#if USE(CG)
    std::optional<SharedMemoryHandle> m_ownershipHandle;
    bool m_releaseBitmapContextDataCalled : 1 { false };
#endif
};

} // namespace WebCore
