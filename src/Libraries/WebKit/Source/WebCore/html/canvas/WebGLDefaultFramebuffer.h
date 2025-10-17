/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#if ENABLE(WEBGL)

#include "WebGLRenderingContextBase.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// Implementation for the WebGL context default framebuffer.
class WebGLDefaultFramebuffer {
    WTF_MAKE_TZONE_ALLOCATED(WebGLDefaultFramebuffer);
    WTF_MAKE_NONCOPYABLE(WebGLDefaultFramebuffer);
public:
    static std::unique_ptr<WebGLDefaultFramebuffer> create(WebGLRenderingContextBase&, IntSize);
    ~WebGLDefaultFramebuffer() = default;

    PlatformGLObject object() const { return 0; }
    bool hasStencil() const { return m_hasStencil; }
    bool hasDepth() const { return m_hasDepth; }
    IntSize size() const;
    void reshape(IntSize);
    GCGLbitfield dirtyBuffers() const { return m_dirtyBuffers; }
    void markBuffersClear(GCGLbitfield clearBuffers);
    void markAllUnpreservedBuffersDirty();
    void markAllBuffersDirty();

private:
    WebGLDefaultFramebuffer(WebGLRenderingContextBase&);

    WeakRef<WebGLRenderingContextBase> m_context;

    GCGLbitfield m_unpreservedBuffers { 0 };
    GCGLbitfield m_dirtyBuffers { 0 };
    bool m_hasStencil : 1;
    bool m_hasDepth : 1;
};

}

#endif
