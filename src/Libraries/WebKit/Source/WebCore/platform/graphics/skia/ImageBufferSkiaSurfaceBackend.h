/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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

#if USE(SKIA)
#include "GraphicsContextSkia.h"
#include "ImageBufferSkiaBackend.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkSurface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/Forward.h>

namespace WebCore {

class ImageBufferSkiaSurfaceBackend : public ImageBufferSkiaBackend {
public:
    virtual ~ImageBufferSkiaSurfaceBackend();

    static IntSize calculateSafeBackendSize(const Parameters&);
    static unsigned calculateBytesPerRow(const IntSize&);
    static size_t calculateMemoryCost(const Parameters&);

protected:
    ImageBufferSkiaSurfaceBackend(const Parameters&, sk_sp<SkSurface>&&, RenderingMode);

    GraphicsContext& context() final { return m_context; }
    unsigned bytesPerRow() const final;
    bool canMapBackingStore() const final;

    sk_sp<SkSurface> m_surface;
    GraphicsContextSkia m_context;
};

} // namespace WebCore

#endif // USE(SKIA)
