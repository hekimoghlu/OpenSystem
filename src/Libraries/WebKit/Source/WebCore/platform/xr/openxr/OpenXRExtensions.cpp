/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
#include "OpenXRExtensions.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEBXR) && USE(OPENXR)

using namespace WebCore;

namespace PlatformXR {

WTF_MAKE_TZONE_ALLOCATED_IMPL(OpenXRExtensions);

std::unique_ptr<OpenXRExtensions> OpenXRExtensions::create()
{
    uint32_t extensionCount { 0 };
    XrResult result = xrEnumerateInstanceExtensionProperties(nullptr, 0, &extensionCount, nullptr);

    if (XR_FAILED(result) || !extensionCount) {
        LOG(XR, "xrEnumerateInstanceExtensionProperties(): no extensions\n");
        return nullptr;
    }

    Vector<XrExtensionProperties> extensions(extensionCount, [] {
        return createStructure<XrExtensionProperties, XR_TYPE_EXTENSION_PROPERTIES>();
    }());

    result = xrEnumerateInstanceExtensionProperties(nullptr, extensionCount, &extensionCount, extensions.data());
    if (XR_FAILED(result)) {
        LOG(XR, "xrEnumerateInstanceExtensionProperties() failed: %d\n", result);
        return nullptr;
    }

    return makeUnique<OpenXRExtensions>(WTFMove(extensions));
}

OpenXRExtensions::OpenXRExtensions(Vector<XrExtensionProperties>&& extensions)
    : m_extensions(WTFMove(extensions))
{
}

void OpenXRExtensions::loadMethods(XrInstance instance)
{
    m_methods.getProcAddressFunc = eglGetProcAddress;
    xrGetInstanceProcAddr(instance, "xrGetOpenGLGraphicsRequirementsKHR", reinterpret_cast<PFN_xrVoidFunction*>(&m_methods.xrGetOpenGLGraphicsRequirementsKHR));
}

bool OpenXRExtensions::isExtensionSupported(const char* name) const
{
    auto position = m_extensions.findIf([name](auto& property) {
        return !strcmp(property.extensionName, name);
    });
    return position != notFound;
}

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
