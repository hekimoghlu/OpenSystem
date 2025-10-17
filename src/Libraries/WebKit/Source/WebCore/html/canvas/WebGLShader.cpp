/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

#include "WebGLShader.h"

#include "WebGLRenderingContextBase.h"
#include <wtf/Lock.h>
#include <wtf/Locker.h>

namespace WebCore {

RefPtr<WebGLShader> WebGLShader::create(WebGLRenderingContextBase& context, GCGLenum type)
{
    auto object = context.protectedGraphicsContextGL()->createShader(type);
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLShader(context, object, type));
}

WebGLShader::WebGLShader(WebGLRenderingContextBase& context, PlatformGLObject object, GCGLenum type)
    : WebGLObject(context, object)
    , m_type(type)
    , m_source(emptyString())
{
}

WebGLShader::~WebGLShader()
{
    if (!m_context)
        return;

    runDestructor();
}

void WebGLShader::deleteObjectImpl(const AbstractLocker&, GraphicsContextGL* context3d, PlatformGLObject object)
{
    context3d->deleteShader(object);
}

}

#endif // ENABLE(WEBGL)
