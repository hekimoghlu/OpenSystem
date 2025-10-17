/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#include "GraphicsContextGLTextureMapperANGLE.h"

#if ENABLE(WEBGL) && USE(TEXTURE_MAPPER) && USE(COORDINATED_GRAPHICS) && USE(LIBEPOXY)
#include <epoxy/gl.h>

namespace WebCore {

GCGLuint GraphicsContextGLTextureMapperANGLE::setupCurrentTexture()
{
    // Current texture was bound by ANGLE, we query using epoxy to get the actual texture ID.
    GLint texture;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture);

    // The texture has been configured by ANGLE too, but the values are cached and only applied
    // when another call causes a texture state sync, which doesn't happen. So, we set the same
    // parmeters here using epoxy to make sure the texture is configured as expected by the
    // texture mapper.
    GLenum textureTarget = drawingBufferTextureTarget();
    glTexParameteri(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

} // namespace WebCore

#endif // ENABLE(WEBGL) && USE(TEXTURE_MAPPER) && USE(COORDINATED_GRAPHICS) && USE(LIBEPOXY)
