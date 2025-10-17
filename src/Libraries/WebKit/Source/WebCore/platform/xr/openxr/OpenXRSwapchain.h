/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

#if ENABLE(WEBXR) && USE(OPENXR)

#include "GraphicsTypesGL.h"
#include "IntSize.h"
#include "OpenXRUtils.h"

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace PlatformXR {

class OpenXRSwapchain {
    WTF_MAKE_TZONE_ALLOCATED(OpenXRSwapchain);
    WTF_MAKE_NONCOPYABLE(OpenXRSwapchain);
public:
    static std::unique_ptr<OpenXRSwapchain> create(XrInstance, XrSession, const XrSwapchainCreateInfo&);
    ~OpenXRSwapchain();

    std::optional<PlatformGLObject> acquireImage();
    void releaseImage();
    XrSwapchain swapchain() const { return m_swapchain; }
    int32_t width() const { return m_createInfo.width; }
    int32_t height() const { return m_createInfo.height; }
    WebCore::IntSize size() const { return WebCore::IntSize(width(), height()); }

private:
    OpenXRSwapchain(XrInstance, XrSwapchain, const XrSwapchainCreateInfo&, Vector<XrSwapchainImageOpenGLKHR>&&);

    XrInstance m_instance;
    XrSwapchain m_swapchain;
    XrSwapchainCreateInfo m_createInfo;
    Vector<XrSwapchainImageOpenGLKHR> m_imageBuffers;
    PlatformGLObject m_acquiredTexture { 0 };
};


} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
