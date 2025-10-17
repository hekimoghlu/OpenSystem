/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#include "WCSceneContext.h"

#if USE(GRAPHICS_LAYER_WC)

#include <WebCore/GLContext.h>
#include <WebCore/TextureMapper.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WCSceneContext);

WCSceneContext::WCSceneContext(uint64_t nativeWindow)
{
    m_glContext = WebCore::GLContext::create(reinterpret_cast<GLNativeWindowType>(nativeWindow), WebCore::PlatformDisplay::sharedDisplay());
}

WCSceneContext::~WCSceneContext() = default;

bool WCSceneContext::makeContextCurrent()
{
    if (!m_glContext)
        return false;
    return m_glContext->makeContextCurrent();
}

std::unique_ptr<WebCore::TextureMapper> WCSceneContext::createTextureMapper()
{
    return WebCore::TextureMapper::create();
}

void WCSceneContext::swapBuffers()
{
    m_glContext->swapBuffers();
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
