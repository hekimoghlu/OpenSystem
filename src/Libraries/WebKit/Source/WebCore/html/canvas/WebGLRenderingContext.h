/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
#include <memory>

namespace WebCore {

class WebGLTimerQueryEXT;

class WebGLRenderingContext final : public WebGLRenderingContextBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebGLRenderingContext);
public:
    static std::unique_ptr<WebGLRenderingContext> create(CanvasBase&, WebGLContextAttributes&&);

    ~WebGLRenderingContext();

    std::optional<WebGLExtensionAny> getExtension(const String&) final;
    std::optional<Vector<String>> getSupportedExtensions() final;

    WebGLAny getFramebufferAttachmentParameter(GCGLenum target, GCGLenum attachment, GCGLenum pname) final;

    long long getInt64Parameter(GCGLenum) final;

    GCGLint maxDrawBuffers() final;
    GCGLint maxColorAttachments() final;
    void initializeDefaultObjects() final;

    void addMembersToOpaqueRoots(JSC::AbstractSlotVisitor&) final;

protected:
    friend class EXTDisjointTimerQuery;

    WebGLBindingPoint<WebGLTimerQueryEXT, GraphicsContextGL::TIME_ELAPSED_EXT> m_activeQuery;

private:
    using WebGLRenderingContextBase::WebGLRenderingContextBase;
};

WebCoreOpaqueRoot root(const WebGLExtension<WebGLRenderingContext>*);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CANVASRENDERINGCONTEXT(WebCore::WebGLRenderingContext, isWebGL1())

#endif
