/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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

#if USE(GRAPHICS_LAYER_WC)

#include <wtf/TZoneMalloc.h>

namespace WebCore {
class GLContext;
class TextureMapper;
}

namespace WebKit {

class WCSceneContext {
    WTF_MAKE_TZONE_ALLOCATED(WCSceneContext);
public:
    explicit WCSceneContext(uint64_t nativeWindow);
    ~WCSceneContext();

    bool makeContextCurrent();
    std::unique_ptr<WebCore::TextureMapper> createTextureMapper();
    void swapBuffers();

private:
    std::unique_ptr<WebCore::GLContext> m_glContext;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
