/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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

#include <JavaScriptCore/Forward.h>

namespace JSC {
class CallFrame;
class JSValue;
}

namespace WebCore {

class JSDOMGlobalObject;
class WebGLBuffer;
class WebGLFramebuffer;
class WebGLProgram;
class WebGLQuery;
class WebGLRenderbuffer;
class WebGLSampler;
class WebGLTexture;
class WebGLTimerQueryEXT;
class WebGLTransformFeedback;
class WebGLVertexArrayObject;
class WebGLVertexArrayObjectOES;

using WebGLAny = std::variant<
    std::nullptr_t,
    bool,
    int,
    unsigned,
    long long,
    unsigned long long,
    float,
    String,
    Vector<bool>,
    Vector<int>,
    Vector<unsigned>,
    RefPtr<Float32Array>,
    RefPtr<Int32Array>,
    RefPtr<Uint32Array>,
    RefPtr<Uint8Array>,
    RefPtr<WebGLBuffer>,
    RefPtr<WebGLFramebuffer>,
    RefPtr<WebGLProgram>,
    RefPtr<WebGLQuery>,
    RefPtr<WebGLRenderbuffer>,
    RefPtr<WebGLSampler>,
    RefPtr<WebGLTexture>,
    RefPtr<WebGLTimerQueryEXT>,
    RefPtr<WebGLTransformFeedback>,
    RefPtr<WebGLVertexArrayObject>,
    RefPtr<WebGLVertexArrayObjectOES>
>;

} // namespace WebCore

#endif
