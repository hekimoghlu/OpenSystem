/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#include "OESVertexArrayObject.h"

#include <wtf/Lock.h>
#include <wtf/Locker.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(OESVertexArrayObject);

OESVertexArrayObject::OESVertexArrayObject(WebGLRenderingContext& context)
    : WebGLExtension(context, WebGLExtensionName::OESVertexArrayObject)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_OES_vertex_array_object"_s);
}

OESVertexArrayObject::~OESVertexArrayObject() = default;

bool OESVertexArrayObject::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_OES_vertex_array_object"_s);
}

RefPtr<WebGLVertexArrayObjectOES> OESVertexArrayObject::createVertexArrayOES()
{
    if (isContextLost())
        return nullptr;
    auto& context = this->context();
    return WebGLVertexArrayObjectOES::createUser(context);
}

void OESVertexArrayObject::deleteVertexArrayOES(WebGLVertexArrayObjectOES* arrayObject)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    Locker locker { context.objectGraphLock() };

    if (!arrayObject)
        return;

    if (!arrayObject->validate(context)) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "delete"_s, "object does not belong to this context"_s);
        return;
    }

    if (arrayObject->isDeleted())
        return;

    if (!arrayObject->isDefaultObject() && arrayObject == context.m_boundVertexArrayObject)
        context.setBoundVertexArrayObject(locker, nullptr);

    arrayObject->deleteObject(locker, context.protectedGraphicsContextGL().get());
}

GCGLboolean OESVertexArrayObject::isVertexArrayOES(WebGLVertexArrayObjectOES* arrayObject)
{
    if (isContextLost())
        return false;
    auto& context = this->context();
    if (!context.validateIsWebGLObject(arrayObject))
        return false;
    return context.protectedGraphicsContextGL()->isVertexArray(arrayObject->object());
}

void OESVertexArrayObject::bindVertexArrayOES(WebGLVertexArrayObjectOES* arrayObject)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    Locker locker { context.objectGraphLock() };

    // Checks for already deleted objects and objects from other contexts. 
    if (!context.validateNullableWebGLObject("bindVertexArrayOES"_s, arrayObject))
        return;

    RefPtr contextGL = context.graphicsContextGL();
    if (arrayObject && !arrayObject->isDefaultObject() && arrayObject->object()) {
        contextGL->bindVertexArray(arrayObject->object());
        context.setBoundVertexArrayObject(locker, arrayObject);
    } else {
        contextGL->bindVertexArray(0);
        context.setBoundVertexArrayObject(locker, nullptr);
    }
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
