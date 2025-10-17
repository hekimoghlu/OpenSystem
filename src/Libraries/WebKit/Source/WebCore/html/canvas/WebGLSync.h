/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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

#include "GraphicsContextGL.h"
#include "WebGLObject.h"

namespace WebCore {

class WebGLSync final : public WebGLObject {
public:
    virtual ~WebGLSync();

    static RefPtr<WebGLSync> create(WebGLRenderingContextBase&);

    void updateCache(WebGLRenderingContextBase&);
    GCGLint getCachedResult(GCGLenum pname) const;
    bool isSignaled() const;
    void scheduleAllowCacheUpdate(WebGLRenderingContextBase&);

    bool isUsable() const { return object() && !isDeleted(); }
    bool isInitialized() const { return true; }
private:
    WebGLSync(WebGLRenderingContextBase&, GCGLsync);
    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override;

    bool m_allowCacheUpdate = { false };
    GCGLint m_syncStatus = { GraphicsContextGL::UNSIGNALED };
    GCGLsync m_sync;
};

} // namespace WebCore

#endif
