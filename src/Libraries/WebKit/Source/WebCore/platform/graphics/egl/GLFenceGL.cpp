/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#include "GLFenceGL.h"

#if HAVE(GL_FENCE)

#if USE(LIBEPOXY)
#include <epoxy/gl.h>
#else
#include <GLES3/gl3.h>
#endif

namespace WebCore {

std::unique_ptr<GLFence> GLFenceGL::create()
{
    if (auto* sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)) {
        glFlush();
        return makeUnique<GLFenceGL>(sync);
    }

    return nullptr;
}

GLFenceGL::GLFenceGL(GLsync sync)
    : m_sync(sync)
{
}

GLFenceGL::~GLFenceGL()
{
    glDeleteSync(m_sync);
}

void GLFenceGL::clientWait()
{
    glClientWaitSync(m_sync, 0, GL_TIMEOUT_IGNORED);
}

void GLFenceGL::serverWait()
{
    glWaitSync(m_sync, 0, GL_TIMEOUT_IGNORED);
}

} // namespace WebCore

#endif // HAVE(GL_FENCE)
