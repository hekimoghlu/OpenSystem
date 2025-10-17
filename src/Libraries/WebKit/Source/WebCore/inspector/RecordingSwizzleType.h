/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

namespace WebCore {

// Keep this in sync with WI.Recording.Swizzle.
enum class RecordingSwizzleType : int {
    None = 0,
    Number = 1,
    Boolean = 2,
    String = 3,
    Array = 4,
    TypedArray = 5,
    Image = 6,
    ImageData = 7,
    DOMMatrix = 8,
    Path2D = 9,
    CanvasGradient = 10,
    CanvasPattern = 11,
    WebGLBuffer = 12,
    WebGLFramebuffer = 13,
    WebGLRenderbuffer = 14,
    WebGLTexture = 15,
    WebGLShader = 16,
    WebGLProgram = 17,
    WebGLUniformLocation = 18,
    ImageBitmap = 19,
    WebGLQuery = 20,
    WebGLSampler = 21,
    WebGLSync = 22,
    WebGLTransformFeedback = 23,
    WebGLVertexArrayObject = 24,
    DOMPointInit = 25,
};

} // namespace WebCore
