/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include "config.h"
#include "AcceleratedSurfaceLibWPE.h"

#if USE(WPE_RENDERER)

#include "WebPage.h"
#include <WebCore/PlatformDisplayLibWPE.h>
#include <wpe/wpe-egl.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UniStdExtras.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(AcceleratedSurfaceLibWPE);

std::unique_ptr<AcceleratedSurfaceLibWPE> AcceleratedSurfaceLibWPE::create(WebPage& webPage, Function<void()>&& frameCompleteHandler)
{
    return std::unique_ptr<AcceleratedSurfaceLibWPE>(new AcceleratedSurfaceLibWPE(webPage, WTFMove(frameCompleteHandler)));
}

AcceleratedSurfaceLibWPE::AcceleratedSurfaceLibWPE(WebPage& webPage, Function<void()>&& frameCompleteHandler)
    : AcceleratedSurface(webPage, WTFMove(frameCompleteHandler))
    , m_hostFD(webPage.hostFileDescriptor())
    , m_initialSize(webPage.size())
{
    m_initialSize.scale(webPage.deviceScaleFactor());
}

AcceleratedSurfaceLibWPE::~AcceleratedSurfaceLibWPE()
{
    ASSERT(!m_backend);
}

void AcceleratedSurfaceLibWPE::finalize()
{
    wpe_renderer_backend_egl_target_destroy(m_backend);
    m_backend = nullptr;
}

void AcceleratedSurfaceLibWPE::initialize()
{
    ASSERT(!m_backend);
    m_backend = wpe_renderer_backend_egl_target_create(m_hostFD.release());
    static struct wpe_renderer_backend_egl_target_client s_client = {
        // frame_complete
        [](void* data)
        {
            auto& surface = *reinterpret_cast<AcceleratedSurfaceLibWPE*>(data);
            surface.frameComplete();
        },
        // padding
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    wpe_renderer_backend_egl_target_set_client(m_backend, &s_client, this);
    wpe_renderer_backend_egl_target_initialize(m_backend, downcast<PlatformDisplayLibWPE>(PlatformDisplay::sharedDisplay()).backend(),
        std::max(1, m_initialSize.width()), std::max(1, m_initialSize.height()));
}

uint64_t AcceleratedSurfaceLibWPE::window() const
{
    const_cast<AcceleratedSurfaceLibWPE*>(this)->initialize();

    // EGLNativeWindowType changes depending on the EGL implementation: reinterpret_cast works
    // for pointers (only if they are 64-bit wide and not for other cases), and static_cast for
    // numeric types (and when needed they get extended to 64-bit) but not for pointers. Using
    // a plain C cast expression in this one instance works in all cases.
    static_assert(sizeof(EGLNativeWindowType) <= sizeof(uint64_t), "EGLNativeWindowType must not be longer than 64 bits.");
    return (uint64_t)wpe_renderer_backend_egl_target_get_native_window(m_backend);
}

uint64_t AcceleratedSurfaceLibWPE::surfaceID() const
{
    return m_webPage->identifier().toUInt64();
}

bool AcceleratedSurfaceLibWPE::resize(const IntSize& size)
{
    if (!AcceleratedSurface::resize(size))
        return false;

    ASSERT(m_backend);
    wpe_renderer_backend_egl_target_resize(m_backend, std::max(1, size.width()), std::max(1, size.height()));
    return true;
}

void AcceleratedSurfaceLibWPE::willRenderFrame()
{
    ASSERT(m_backend);
    wpe_renderer_backend_egl_target_frame_will_render(m_backend);
}

void AcceleratedSurfaceLibWPE::didRenderFrame()
{
    ASSERT(m_backend);
    wpe_renderer_backend_egl_target_frame_rendered(m_backend);
}

} // namespace WebKit

#endif // USE(WPE_RENDERER)
