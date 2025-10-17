/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#include <atomic>
#include <wtf/RefCounted.h>
namespace WebCore {

class WebCoreOpaqueRoot;

// Manual variant discriminator for any WebGL extension. Used for downcasting the WebGLExtensionBase pointers
// to the concrete type.
enum class WebGLExtensionName {
    ANGLEInstancedArrays,
    EXTBlendMinMax,
    EXTClipControl,
    EXTColorBufferFloat,
    EXTColorBufferHalfFloat,
    EXTConservativeDepth,
    EXTDepthClamp,
    EXTDisjointTimerQuery,
    EXTDisjointTimerQueryWebGL2,
    EXTFloatBlend,
    EXTFragDepth,
    EXTPolygonOffsetClamp,
    EXTRenderSnorm,
    EXTShaderTextureLOD,
    EXTTextureCompressionBPTC,
    EXTTextureCompressionRGTC,
    EXTTextureFilterAnisotropic,
    EXTTextureMirrorClampToEdge,
    EXTTextureNorm16,
    EXTsRGB,
    KHRParallelShaderCompile,
    NVShaderNoperspectiveInterpolation,
    OESDrawBuffersIndexed,
    OESElementIndexUint,
    OESFBORenderMipmap,
    OESSampleVariables,
    OESShaderMultisampleInterpolation,
    OESStandardDerivatives,
    OESTextureFloat,
    OESTextureFloatLinear,
    OESTextureHalfFloat,
    OESTextureHalfFloatLinear,
    OESVertexArrayObject,
    WebGLBlendFuncExtended,
    WebGLClipCullDistance,
    WebGLColorBufferFloat,
    WebGLCompressedTextureASTC,
    WebGLCompressedTextureETC,
    WebGLCompressedTextureETC1,
    WebGLCompressedTexturePVRTC,
    WebGLCompressedTextureS3TC,
    WebGLCompressedTextureS3TCsRGB,
    WebGLDebugRendererInfo,
    WebGLDebugShaders,
    WebGLDepthTexture,
    WebGLDrawBuffers,
    WebGLDrawInstancedBaseVertexBaseInstance,
    WebGLLoseContext,
    WebGLMultiDraw,
    WebGLMultiDrawInstancedBaseVertexBaseInstance,
    WebGLPolygonMode,
    WebGLProvokingVertex,
    WebGLRenderSharedExponent,
    WebGLStencilTexturing
};

class WebGLExtensionBase : public RefCounted<WebGLExtensionBase> {
public:
    WebGLExtensionName name() const { return m_name; }

    virtual ~WebGLExtensionBase() = default;

protected:
    WebGLExtensionBase(WebGLExtensionName name)
        : m_name(name)
    {
    }
protected:
    const WebGLExtensionName m_name;
};

// Mixin class for WebGL extension implementations.
// All functions should start with preamble:
// if (isContextLost())
//     return;
// auto& context = this->context();
// context.drawSomething(...);
template<typename T>
class WebGLExtension : public WebGLExtensionBase {
public:
    void loseParentContext() { m_context = nullptr; }
    T& context() { ASSERT(!isContextLost()); return *m_context.load(std::memory_order::relaxed); }

    // Only to be used by friend WebCoreOpaqueRoot root(const WebGLExtension<T>*) that cannot be a friend
    // due to C++ warning on some compilers.
    T* opaqueRoot() const { return m_context.load(); }

protected:
    WebGLExtension(T& context, WebGLExtensionName name)
        : WebGLExtensionBase(name)
        , m_context(&context)
    {
    }
    bool isContextLost() const { return !m_context.load(std::memory_order::relaxed); }

private:
    std::atomic<T*> m_context;
};

} // namespace WebCore

#endif
