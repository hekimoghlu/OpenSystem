/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

#if OS(UNIX)
#include <wtf/unix/UnixFileDescriptor.h>
#endif

typedef struct __GLsync* GLsync;

namespace WebCore {

class GLFence {
    WTF_MAKE_TZONE_ALLOCATED(GLFence);
    WTF_MAKE_NONCOPYABLE(GLFence);
public:
    static bool isSupported();
    WEBCORE_EXPORT static std::unique_ptr<GLFence> create();
#if OS(UNIX)
    WEBCORE_EXPORT static std::unique_ptr<GLFence> createExportable();
    WEBCORE_EXPORT static std::unique_ptr<GLFence> importFD(WTF::UnixFileDescriptor&&);
#endif
    virtual ~GLFence() = default;

    virtual void clientWait() = 0;
    virtual void serverWait() = 0;
#if OS(UNIX)
    virtual WTF::UnixFileDescriptor exportFD() { return { }; }
#endif

protected:
    GLFence() = default;

    struct Capabilities {
        bool eglSupported { false };
        bool eglServerWaitSupported { false };
        bool eglExportableSupported { false };
        bool glSupported { false };
    };
    static const Capabilities& capabilities();
};

} // namespace WebCore
