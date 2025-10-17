/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "WebGLBuffer.h"

#if ENABLE(WEBGL)

#include "WebGLRenderingContextBase.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/Lock.h>
#include <wtf/Locker.h>

namespace WebCore {

RefPtr<WebGLBuffer> WebGLBuffer::create(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->createBuffer();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLBuffer { context, object });
}

WebGLBuffer::WebGLBuffer(WebGLRenderingContextBase& context, PlatformGLObject object)
    : WebGLObject(context, object)
{
}

WebGLBuffer::~WebGLBuffer()
{
    if (!m_context)
        return;

    runDestructor();
}

void WebGLBuffer::deleteObjectImpl(const AbstractLocker&, GraphicsContextGL* context3d, PlatformGLObject object)
{
    context3d->deleteBuffer(object);
}

}

#endif // ENABLE(WEBGL)
