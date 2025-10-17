/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "AcceleratedSurface.h"

#include "WebPage.h"
#include <WebCore/PlatformDisplay.h>
#include <wtf/TZoneMallocInlines.h>

#if USE(WPE_RENDERER)
#include "AcceleratedSurfaceLibWPE.h"
#endif

#if (PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM)))
#include "AcceleratedSurfaceDMABuf.h"
#endif

#if USE(LIBEPOXY)
#include <epoxy/gl.h>
#else
#include <GLES2/gl2.h>
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(AcceleratedSurface);

std::unique_ptr<AcceleratedSurface> AcceleratedSurface::create(ThreadedCompositor& compositor, WebPage& webPage, Function<void()>&& frameCompleteHandler)
{
#if (PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM)))
#if USE(GBM)
    if (PlatformDisplay::sharedDisplay().type() == PlatformDisplay::Type::GBM)
        return AcceleratedSurfaceDMABuf::create(compositor, webPage, WTFMove(frameCompleteHandler));
#endif
    if (PlatformDisplay::sharedDisplay().type() == PlatformDisplay::Type::Surfaceless)
        return AcceleratedSurfaceDMABuf::create(compositor, webPage, WTFMove(frameCompleteHandler));
#endif
#if USE(WPE_RENDERER)
    if (PlatformDisplay::sharedDisplay().type() == PlatformDisplay::Type::WPE)
        return AcceleratedSurfaceLibWPE::create(webPage, WTFMove(frameCompleteHandler));
#endif
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

AcceleratedSurface::AcceleratedSurface(WebPage& webPage, Function<void()>&& frameCompleteHandler)
    : m_webPage(webPage)
    , m_frameCompleteHandler(WTFMove(frameCompleteHandler))
    , m_isOpaque(!webPage.backgroundColor().has_value() || webPage.backgroundColor()->isOpaque())
{
}

bool AcceleratedSurface::resize(const IntSize& size)
{
    if (m_size == size)
        return false;

    m_size = size;
    return true;
}

bool AcceleratedSurface::backgroundColorDidChange()
{
    const auto& color = m_webPage->backgroundColor();
    auto isOpaque = !color.has_value() || color->isOpaque();
    if (m_isOpaque == isOpaque)
        return false;

    m_isOpaque = isOpaque;
    return true;
}

void AcceleratedSurface::clearIfNeeded()
{
    if (m_isOpaque)
        return;

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
}

void AcceleratedSurface::frameComplete() const
{
    m_frameCompleteHandler();
}

} // namespace WebKit
