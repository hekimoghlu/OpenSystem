/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

#if ENABLE(WEBXR) && USE(OPENXR)
#include "OpenXRSwapchain.h"
#include <wtf/TZoneMallocInlines.h>

using namespace WebCore;

namespace PlatformXR {

WTF_MAKE_TZONE_ALLOCATED_IMPL(OpenXRSwapchain);

std::unique_ptr<OpenXRSwapchain> OpenXRSwapchain::create(XrInstance instance, XrSession session, const XrSwapchainCreateInfo& info)
{
    ASSERT(session != XR_NULL_HANDLE);
    ASSERT(info.faceCount == 1);

    XrSwapchain swapchain { XR_NULL_HANDLE };
    auto result = xrCreateSwapchain(session, &info, &swapchain);
    RETURN_IF_FAILED(result, "xrEnumerateInstanceExtensionProperties", instance, nullptr);
    ASSERT(swapchain != XR_NULL_HANDLE);

    uint32_t imageCount;
    result = xrEnumerateSwapchainImages(swapchain, 0, &imageCount, nullptr);
    RETURN_IF_FAILED(result, "xrEnumerateSwapchainImages", instance, nullptr);
    if (!imageCount) {
        LOG(XR, "xrEnumerateSwapchainImages(): no images\n");
        return nullptr;
    }

    Vector<XrSwapchainImageOpenGLKHR> imageBuffers(imageCount, [] {
        return createStructure<XrSwapchainImageOpenGLKHR, XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR>();
    }());

    Vector<XrSwapchainImageBaseHeader*> imageHeaders = imageBuffers.map([](auto& image) {
        return (XrSwapchainImageBaseHeader*) &image;
    });

    // Get images from an XrSwapchain
    result = xrEnumerateSwapchainImages(swapchain, imageCount, &imageCount, imageHeaders[0]);
    RETURN_IF_FAILED(result, "xrEnumerateSwapchainImages with imageCount", instance, nullptr);

    return std::unique_ptr<OpenXRSwapchain>(new OpenXRSwapchain(instance, swapchain, info, WTFMove(imageBuffers)));
}

OpenXRSwapchain::OpenXRSwapchain(XrInstance instance, XrSwapchain swapchain, const XrSwapchainCreateInfo& info, Vector<XrSwapchainImageOpenGLKHR>&& imageBuffers)
    : m_instance(instance)
    , m_swapchain(swapchain)
    , m_createInfo(info)
    , m_imageBuffers(WTFMove(imageBuffers))
{
}

OpenXRSwapchain::~OpenXRSwapchain()
{
    if (m_acquiredTexture)
        releaseImage();
    if (m_swapchain != XR_NULL_HANDLE)
        xrDestroySwapchain(m_swapchain);
}

std::optional<PlatformGLObject> OpenXRSwapchain::acquireImage()
{
#if LOG_DISABLED
    UNUSED_VARIABLE(m_instance);
#endif

    RELEASE_ASSERT_WITH_MESSAGE(!m_acquiredTexture , "Expected no acquired images. ReleaseImage not called?");

    auto acquireInfo = createStructure<XrSwapchainImageAcquireInfo, XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO>();
    uint32_t swapchainImageIndex = 0;
    auto result = xrAcquireSwapchainImage(m_swapchain, &acquireInfo, &swapchainImageIndex);
    RETURN_IF_FAILED(result, "xrAcquireSwapchainImage", m_instance, std::nullopt);

    RELEASE_ASSERT(swapchainImageIndex < m_imageBuffers.size());

    auto waitInfo = createStructure<XrSwapchainImageWaitInfo, XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO>();
    waitInfo.timeout = XR_INFINITE_DURATION;
    result = xrWaitSwapchainImage(m_swapchain, &waitInfo);
    RETURN_IF_FAILED(result, "xrWaitSwapchainImage", m_instance, std::nullopt);

    m_acquiredTexture = m_imageBuffers[swapchainImageIndex].image;

    return m_acquiredTexture;
}

void OpenXRSwapchain::releaseImage()
{
    RELEASE_ASSERT_WITH_MESSAGE(m_acquiredTexture, "Expected a valid acquired image. AcquireImage not called?");

    auto releaseInfo = createStructure<XrSwapchainImageReleaseInfo, XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO>();
    auto result = xrReleaseSwapchainImage(m_swapchain, &releaseInfo);
    LOG_IF_FAILED(result, "xrReleaseSwapchainImage", m_instance);

    m_acquiredTexture = 0;
}

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
