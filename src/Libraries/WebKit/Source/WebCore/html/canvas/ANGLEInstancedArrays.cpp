/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#include "ANGLEInstancedArrays.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ANGLEInstancedArrays);

ANGLEInstancedArrays::ANGLEInstancedArrays(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::ANGLEInstancedArrays)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_ANGLE_instanced_arrays"_s);
}

ANGLEInstancedArrays::~ANGLEInstancedArrays() = default;

bool ANGLEInstancedArrays::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_ANGLE_instanced_arrays"_s);
}

void ANGLEInstancedArrays::drawArraysInstancedANGLE(GCGLenum mode, GCGLint first, GCGLsizei count, GCGLsizei primcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.drawArraysInstanced(mode, first, count, primcount);
}

void ANGLEInstancedArrays::drawElementsInstancedANGLE(GCGLenum mode, GCGLsizei count, GCGLenum type, long long offset, GCGLsizei primcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.drawElementsInstanced(mode, count, type, offset, primcount);
}

void ANGLEInstancedArrays::vertexAttribDivisorANGLE(GCGLuint index, GCGLuint divisor)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.vertexAttribDivisor(index, divisor);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
