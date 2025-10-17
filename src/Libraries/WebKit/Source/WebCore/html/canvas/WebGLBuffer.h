/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include "WebGLObject.h"
#include <wtf/RefPtr.h>

namespace JSC {
class ArrayBuffer;
class ArrayBufferView;
}

namespace WebCore {

class WebGLBuffer final : public WebGLObject {
public:
    static RefPtr<WebGLBuffer> create(WebGLRenderingContextBase&);
    virtual ~WebGLBuffer();

    GCGLenum getTarget() const { return m_target; }
    void didBind(GCGLenum target);
    bool isUsable() const { return object() && !isDeleted(); }
    bool isInitialized() const { return m_target; }
private:
    WebGLBuffer(WebGLRenderingContextBase&, PlatformGLObject);

    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override;

    GCGLenum m_target { 0 };
};

inline void WebGLBuffer::didBind(GCGLenum target)
{
    if (m_target)
        return;
    m_target = target;
}

} // namespace WebCore

#endif
