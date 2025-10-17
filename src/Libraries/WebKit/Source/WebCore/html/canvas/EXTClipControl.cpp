/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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

#if ENABLE(WEBGL)
#include "EXTClipControl.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(EXTClipControl);

EXTClipControl::EXTClipControl(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::EXTClipControl)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_EXT_clip_control"_s);
}

EXTClipControl::~EXTClipControl() = default;

bool EXTClipControl::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_EXT_clip_control"_s);
}

void EXTClipControl::clipControlEXT(GCGLenum origin, GCGLenum depth)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->clipControlEXT(origin, depth);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
