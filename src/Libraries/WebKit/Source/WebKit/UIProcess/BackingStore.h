/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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

#if !PLATFORM(WPE)

#include <WebCore/IntSize.h>
#include <WebCore/PlatformImage.h>
#include <pal/HysteresisActivity.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMallocInlines.h>

#if USE(CAIRO) || PLATFORM(GTK)
#include <WebCore/RefPtrCairo.h>
#elif USE(SKIA)
class SkCanvas;
IGNORE_CLANG_WARNINGS_BEGIN("cast-align")
#include <skia/core/SkSurface.h>
IGNORE_CLANG_WARNINGS_END
#endif

namespace WebCore {
class IntRect;
}

namespace WebKit {
struct UpdateInfo;

#if USE(CAIRO) || PLATFORM(GTK)
typedef struct _cairo cairo_t;
using PlatformPaintContextPtr = cairo_t*;
#elif USE(SKIA)
using PlatformPaintContextPtr = SkCanvas*;
#endif

class BackingStore {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(BackingStore);
    WTF_MAKE_NONCOPYABLE(BackingStore);
public:
    BackingStore(const WebCore::IntSize&, float deviceScaleFactor);
    ~BackingStore();

    const WebCore::IntSize& size() const { return m_size; }
    float deviceScaleFactor() const { return m_deviceScaleFactor; }

    void paint(PlatformPaintContextPtr, const WebCore::IntRect&);
    void incorporateUpdate(UpdateInfo&&);

private:
    void scroll(const WebCore::IntRect&, const WebCore::IntSize&);

    WebCore::IntSize m_size;
    float m_deviceScaleFactor { 1 };
#if PLATFORM(GTK) || USE(CAIRO)
    RefPtr<cairo_surface_t> m_surface;
    RefPtr<cairo_surface_t> m_scrollSurface;
    PAL::HysteresisActivity m_scrolledHysteresis;
#elif USE(SKIA)
    sk_sp<SkSurface> m_surface;
#endif
};

} // namespace WebKit

#endif // !PLATFORM(WPE)
