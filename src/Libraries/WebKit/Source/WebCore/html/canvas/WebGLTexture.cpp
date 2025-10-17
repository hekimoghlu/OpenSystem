/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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

#include "WebGLTexture.h"

#include "WebGLFramebuffer.h"
#include "WebGLRenderingContextBase.h"

namespace WebCore {

RefPtr<WebGLTexture> WebGLTexture::create(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->createTexture();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLTexture { context, object });
}

WebGLTexture::WebGLTexture(WebGLRenderingContextBase& context, PlatformGLObject object)
    : WebGLObject(context, object)
{
}

WebGLTexture::~WebGLTexture()
{
    if (!m_context)
        return;

    runDestructor();
}

void WebGLTexture::didBind(GCGLenum target)
{
    if (!object())
        return;
    // Target is finalized the first time bindTexture() is called.
    if (m_target)
        return;
    m_target = target;
}

void WebGLTexture::deleteObjectImpl(const AbstractLocker&, GraphicsContextGL* context3d, PlatformGLObject object)
{
    context3d->deleteTexture(object);
}

GCGLint WebGLTexture::computeLevelCount(GCGLsizei width, GCGLsizei height)
{
    // return 1 + log2Floor(std::max(width, height));
    GCGLsizei n = std::max(width, height);
    if (n <= 0)
        return 0;
    GCGLint log = 0;
    GCGLsizei value = n;
    for (int ii = 4; ii >= 0; --ii) {
        int shift = (1 << ii);
        GCGLsizei x = (value >> shift);
        if (x) {
            value = x;
            log += shift;
        }
    }
    ASSERT(value == 1);
    return log + 1;
}

}

#endif // ENABLE(WEBGL)
