/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#if USE(WPE_RENDERER)

#include "AcceleratedSurface.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/unix/UnixFileDescriptor.h>

struct wpe_renderer_backend_egl_target;

namespace WebKit {

class WebPage;

class AcceleratedSurfaceLibWPE final : public AcceleratedSurface {
    WTF_MAKE_NONCOPYABLE(AcceleratedSurfaceLibWPE);
    WTF_MAKE_TZONE_ALLOCATED(AcceleratedSurfaceLibWPE);
public:
    static std::unique_ptr<AcceleratedSurfaceLibWPE> create(WebPage&, Function<void()>&& frameCompleteHandler);
    ~AcceleratedSurfaceLibWPE();

    uint64_t window() const override;
    uint64_t surfaceID() const override;
    bool resize(const WebCore::IntSize&) override;
    void finalize() override;
    void willRenderFrame() override;
    void didRenderFrame() override;

private:
    AcceleratedSurfaceLibWPE(WebPage&, Function<void()>&& frameCompleteHandler);

    void initialize();

    UnixFileDescriptor m_hostFD;
    WebCore::IntSize m_initialSize;
    struct wpe_renderer_backend_egl_target* m_backend { nullptr };
};

} // namespace WebKit

#endif // USE(WPE_RENDERER)
