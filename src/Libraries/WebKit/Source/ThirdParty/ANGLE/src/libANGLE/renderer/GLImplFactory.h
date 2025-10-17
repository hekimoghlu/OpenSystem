/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GLImplFactory.h:
//   Factory interface for OpenGL ES Impl objects.
//

#ifndef LIBANGLE_RENDERER_GLIMPLFACTORY_H_
#define LIBANGLE_RENDERER_GLIMPLFACTORY_H_

#include <vector>

#include "angle_gl.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/Overlay.h"
#include "libANGLE/Program.h"
#include "libANGLE/ProgramExecutable.h"
#include "libANGLE/ProgramPipeline.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Shader.h"
#include "libANGLE/Texture.h"
#include "libANGLE/TransformFeedback.h"
#include "libANGLE/VertexArray.h"
#include "libANGLE/renderer/serial_utils.h"

namespace gl
{
class State;
}  // namespace gl

namespace rx
{
class BufferImpl;
class CompilerImpl;
class ContextImpl;
class FenceNVImpl;
class SyncImpl;
class FramebufferImpl;
class MemoryObjectImpl;
class OverlayImpl;
class PathImpl;
class ProgramExecutableImpl;
class ProgramImpl;
class ProgramPipelineImpl;
class QueryImpl;
class RenderbufferImpl;
class SamplerImpl;
class SemaphoreImpl;
class ShaderImpl;
class TextureImpl;
class TransformFeedbackImpl;
class VertexArrayImpl;

class GLImplFactory : angle::NonCopyable
{
  public:
    GLImplFactory();
    virtual ~GLImplFactory();

    // Shader creation
    virtual CompilerImpl *createCompiler()                           = 0;
    virtual ShaderImpl *createShader(const gl::ShaderState &data)    = 0;
    virtual ProgramImpl *createProgram(const gl::ProgramState &data) = 0;
    virtual ProgramExecutableImpl *createProgramExecutable(
        const gl::ProgramExecutable *executable) = 0;

    // Framebuffer creation
    virtual FramebufferImpl *createFramebuffer(const gl::FramebufferState &data) = 0;

    // Texture creation
    virtual TextureImpl *createTexture(const gl::TextureState &state) = 0;

    // Renderbuffer creation
    virtual RenderbufferImpl *createRenderbuffer(const gl::RenderbufferState &state) = 0;

    // Buffer creation
    virtual BufferImpl *createBuffer(const gl::BufferState &state) = 0;

    // Vertex Array creation
    virtual VertexArrayImpl *createVertexArray(const gl::VertexArrayState &data) = 0;

    // Query and Fence creation
    virtual QueryImpl *createQuery(gl::QueryType type) = 0;
    virtual FenceNVImpl *createFenceNV()               = 0;
    virtual SyncImpl *createSync()                     = 0;

    // Transform Feedback creation
    virtual TransformFeedbackImpl *createTransformFeedback(
        const gl::TransformFeedbackState &state) = 0;

    // Sampler object creation
    virtual SamplerImpl *createSampler(const gl::SamplerState &state) = 0;

    // Program Pipeline object creation
    virtual ProgramPipelineImpl *createProgramPipeline(const gl::ProgramPipelineState &data) = 0;

    // Memory object creation
    virtual MemoryObjectImpl *createMemoryObject() = 0;

    // Semaphore creation
    virtual SemaphoreImpl *createSemaphore() = 0;

    // Overlay creation
    virtual OverlayImpl *createOverlay(const gl::OverlayState &state) = 0;

    rx::UniqueSerial generateSerial() { return mSerialFactory.generate(); }

  private:
    rx::UniqueSerialFactory mSerialFactory;
};

inline GLImplFactory::GLImplFactory() = default;

inline GLImplFactory::~GLImplFactory() = default;

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GLIMPLFACTORY_H_
