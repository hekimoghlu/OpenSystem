/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include "GraphicsContextGL.h"
#include "WebGLObject.h"

namespace WebCore {

class WebGLRenderbuffer final : public WebGLObject {
public:
    virtual ~WebGLRenderbuffer();

    static RefPtr<WebGLRenderbuffer> create(WebGLRenderingContextBase&);

    void setInternalFormat(GCGLenum internalformat)
    {
        m_internalFormat = internalformat;
    }
    GCGLenum getInternalFormat() const { return m_internalFormat; }

    void setSize(GCGLsizei width, GCGLsizei height)
    {
        m_width = width;
        m_height = height;
    }
    GCGLsizei getWidth() const { return m_width; }
    GCGLsizei getHeight() const { return m_height; }

    void setIsValid(bool isValid) { m_isValid = isValid; }
    bool isValid() const { return m_isValid; }

    void didBind() { m_hasEverBeenBound = true; }
    bool hasEverBeenBound() const { return m_hasEverBeenBound; }

    bool isUsable() const { return object() && !isDeleted(); }
    bool isInitialized() const { return m_hasEverBeenBound; }

private:
    WebGLRenderbuffer(WebGLRenderingContextBase&, PlatformGLObject);

    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override;

    GCGLenum m_internalFormat { GraphicsContextGL::RGBA4 };
    GCGLsizei m_width { 0 };
    GCGLsizei m_height { 0 };
    bool m_isValid { true }; // This is only false if internalFormat is DEPTH_STENCIL and packed_depth_stencil is not supported.
    bool m_hasEverBeenBound { false };
};

} // namespace WebCore

#endif
