/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

#if HAVE(GL_FENCE)

#include "GLFence.h"

typedef void* EGLSync;

namespace WebCore {

class GLFenceEGL final : public GLFence {
public:
    static std::unique_ptr<GLFence> create();
#if OS(UNIX)
    static std::unique_ptr<GLFence> createExportable();
    static std::unique_ptr<GLFence> importFD(WTF::UnixFileDescriptor&&);
#endif
    GLFenceEGL(EGLSync, bool);
    virtual ~GLFenceEGL();

private:
    void clientWait() override;
    void serverWait() override;
#if OS(UNIX)
    WTF::UnixFileDescriptor exportFD() override;
#endif

    EGLSync m_sync { nullptr };
    bool m_isExportable { false };
};

} // namespace WebCore

#endif // HAVE(GL_FENCE)
