/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

#include <WebCore/Color.h>
#include <WebCore/IntPoint.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/ListHashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/text/WTFString.h>

#if HAVE(IOSURFACE)
#include <WebCore/IOSurface.h>
#endif

#if PLATFORM(GTK)
#if USE(GTK4)
#include <gtk/gtk.h>
#else
#include <WebCore/RefPtrCairo.h>
#endif
#endif

namespace WebKit {

class WebBackForwardListItem;
class WebPageProxy;

enum class ForceSoftwareCapturingViewportSnapshot : bool { No, Yes };

class ViewSnapshot : public RefCountedAndCanMakeWeakPtr<ViewSnapshot> {
public:
#if HAVE(IOSURFACE)
    static Ref<ViewSnapshot> create(std::unique_ptr<WebCore::IOSurface>);
#endif
#if PLATFORM(GTK)
#if USE(GTK4)
    static Ref<ViewSnapshot> create(GRefPtr<GdkTexture>&&);
#else
    static Ref<ViewSnapshot> create(RefPtr<cairo_surface_t>&&);
#endif
#endif

    ~ViewSnapshot();

    void clearImage();
    bool hasImage() const;

#if HAVE(IOSURFACE)
    id asLayerContents();
    RetainPtr<CGImageRef> asImageForTesting();
#endif

    void setRenderTreeSize(uint64_t renderTreeSize) { m_renderTreeSize = renderTreeSize; }
    uint64_t renderTreeSize() const { return m_renderTreeSize; }

    void setBackgroundColor(const WebCore::Color& color) { m_backgroundColor = color; }
    WebCore::Color backgroundColor() const { return m_backgroundColor; }

    void setViewScrollPosition(WebCore::IntPoint scrollPosition) { m_viewScrollPosition = scrollPosition; }
    WebCore::IntPoint viewScrollPosition() const { return m_viewScrollPosition; }

    void setDeviceScaleFactor(float deviceScaleFactor) { m_deviceScaleFactor = deviceScaleFactor; }
    float deviceScaleFactor() const { return m_deviceScaleFactor; }

    void setOrigin(WebCore::SecurityOriginData&& origin) { m_origin = WTFMove(origin); }
    const WebCore::SecurityOriginData& origin() const { return m_origin; }

#if HAVE(IOSURFACE)
    WebCore::IOSurface* surface() const { return m_surface.get(); }

    size_t estimatedImageSizeInBytes() const { return m_surface ? m_surface->totalBytes() : 0; }
    WebCore::IntSize size() const { return m_surface ? m_surface->size() : WebCore::IntSize(); }

    void setSurface(std::unique_ptr<WebCore::IOSurface>);

    WebCore::SetNonVolatileResult setVolatile(bool);
#endif

#if PLATFORM(GTK)
#if USE(GTK4)
    GdkTexture* texture() const { return m_texture.get(); }
#else
    cairo_surface_t* surface() const { return m_surface.get(); }
#endif

    size_t estimatedImageSizeInBytes() const;
    WebCore::IntSize size() const;
#endif

private:
#if HAVE(IOSURFACE)
    explicit ViewSnapshot(std::unique_ptr<WebCore::IOSurface>);

    std::unique_ptr<WebCore::IOSurface> m_surface;
#endif

#if PLATFORM(GTK)
#if USE(GTK4)
    explicit ViewSnapshot(GRefPtr<GdkTexture>&&);

    GRefPtr<GdkTexture> m_texture;
#else
    explicit ViewSnapshot(RefPtr<cairo_surface_t>&&);

    RefPtr<cairo_surface_t> m_surface;
#endif
#endif

    uint64_t m_renderTreeSize;
    float m_deviceScaleFactor;
    WebCore::Color m_backgroundColor;
    WebCore::IntPoint m_viewScrollPosition; // Scroll position at snapshot time. Integral to make comparison reliable.
    WebCore::SecurityOriginData m_origin;
};

class ViewSnapshotStore {
    WTF_MAKE_NONCOPYABLE(ViewSnapshotStore);
    friend class ViewSnapshot;
public:
    ViewSnapshotStore();
    ~ViewSnapshotStore();

    static ViewSnapshotStore& singleton();

    void recordSnapshot(WebPageProxy&, WebBackForwardListItem&);

    void discardSnapshotImages();
    void discardSnapshotImagesForOrigin(const WebCore::SecurityOriginData&);

    void setDisableSnapshotVolatilityForTesting(bool disable) { m_disableSnapshotVolatility = disable; }
    bool disableSnapshotVolatilityForTesting() const { return m_disableSnapshotVolatility; }

private:
    void didAddImageToSnapshot(ViewSnapshot&);
    void willRemoveImageFromSnapshot(ViewSnapshot&);
    void pruneSnapshots(WebPageProxy&);

    size_t m_snapshotCacheSize { 0 };

    ListHashSet<WeakRef<ViewSnapshot>> m_snapshotsWithImages;
    bool m_disableSnapshotVolatility { false };
};

} // namespace WebKit
