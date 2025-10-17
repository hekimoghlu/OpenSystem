/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#if USE(APPKIT)
OBJC_CLASS NSImage;
#endif

#if ENABLE(MULTI_REPRESENTATION_HEIC)
OBJC_CLASS NSAdaptiveImageGlyph;
#endif

#if USE(CG)
struct CGContext;
#endif

#if PLATFORM(GTK)
#include <wtf/glib/GRefPtr.h>
typedef struct _GdkPixbuf GdkPixbuf;
#if USE(GTK4)
typedef struct _GdkTexture GdkTexture;
#endif
#endif

#if PLATFORM(WIN)
typedef struct tagSIZE SIZE;
typedef SIZE* LPSIZE;
typedef struct HBITMAP__ *HBITMAP;
#endif

#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

#if USE(CG) || USE(APPKIT)
#include <wtf/RetainPtr.h>
#endif

namespace WebCore {

class Image;
class IntSize;
class NativeImage;

class ImageAdapter {
    WTF_MAKE_TZONE_ALLOCATED(ImageAdapter);
public:
    ImageAdapter(Image& image)
        : m_image(image)
    {
    }

    WEBCORE_EXPORT static Ref<Image> loadPlatformResource(const char* name);
#if PLATFORM(WIN)
    WEBCORE_EXPORT static RefPtr<NativeImage> nativeImageOfHBITMAP(HBITMAP);
#endif

#if USE(APPKIT)
    WEBCORE_EXPORT NSImage *nsImage();
    WEBCORE_EXPORT RetainPtr<NSImage> snapshotNSImage();
#endif

#if PLATFORM(COCOA)
    WEBCORE_EXPORT CFDataRef tiffRepresentation();
#endif

#if ENABLE(MULTI_REPRESENTATION_HEIC)
    NSAdaptiveImageGlyph *multiRepresentationHEIC();
#endif

#if PLATFORM(GTK)
    GRefPtr<GdkPixbuf> gdkPixbuf();
#if USE(GTK4)
    GRefPtr<GdkTexture> gdkTexture();
#endif
#endif

#if PLATFORM(WIN)
    WEBCORE_EXPORT bool getHBITMAP(HBITMAP);
    WEBCORE_EXPORT bool getHBITMAPOfSize(HBITMAP, const IntSize*);
#endif
    void invalidate();

#if PLATFORM(COCOA)
    WEBCORE_EXPORT static RetainPtr<CFDataRef> tiffRepresentation(const Vector<Ref<NativeImage>>&);
#endif

private:
    Image& image() const { return m_image.get(); }

    RefPtr<NativeImage> nativeImageOfSize(const IntSize&);
    Vector<Ref<NativeImage>> allNativeImages();

    WeakRef<Image> m_image;

#if USE(APPKIT)
    mutable RetainPtr<NSImage> m_nsImage; // A cached NSImage of all the frames. Only built lazily if someone actually queries for one.
#endif
#if USE(CG)
    mutable RetainPtr<CFDataRef> m_tiffRep; // Cached TIFF rep for all the frames. Only built lazily if someone queries for one.
#endif
#if ENABLE(MULTI_REPRESENTATION_HEIC)
    mutable RetainPtr<NSAdaptiveImageGlyph> m_multiRepHEIC;
#endif
};

} // namespace WebCore
