/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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

#if ENABLE(WEBGL)

#include "WebGLVertexArrayObjectOES.h"

#include "WebGLRenderingContextBase.h"

namespace WebCore {

Ref<WebGLVertexArrayObjectOES> WebGLVertexArrayObjectOES::createDefault(WebGLRenderingContextBase& context)
{
    return adoptRef(*new WebGLVertexArrayObjectOES(context, 0, Type::Default));
}

RefPtr<WebGLVertexArrayObjectOES> WebGLVertexArrayObjectOES::createUser(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->createVertexArray();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLVertexArrayObjectOES { context, object, Type::User });
}

WebGLVertexArrayObjectOES::WebGLVertexArrayObjectOES(WebGLRenderingContextBase& context, PlatformGLObject object, Type type)
    : WebGLVertexArrayObjectBase(context, object, type)
{
}

WebGLVertexArrayObjectOES::~WebGLVertexArrayObjectOES()
{
    if (!context())
        return;

    runDestructor();
}

void WebGLVertexArrayObjectOES::deleteObjectImpl(const AbstractLocker& locker, GraphicsContextGL* context3d, PlatformGLObject object)
{
    switch (m_type) {
    case Type::Default:
        break;
    case Type::User:
        context3d->deleteVertexArray(object);
        break;
    }

    if (m_boundElementArrayBuffer)
        m_boundElementArrayBuffer->onDetached(locker, context3d);

    for (auto& state : m_vertexAttribState) {
        if (state.bufferBinding)
            state.bufferBinding->onDetached(locker, context3d);
    }
}
}

#endif // ENABLE(WEBGL)
