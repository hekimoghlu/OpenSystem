/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#include "OpenXRExtensions.h"
#include "PlatformXROpenXR.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/Scope.h>
#include <wtf/StdLibExtras.h>

using namespace WebCore;

namespace PlatformXR {

struct Instance::Impl {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    Impl();
    ~Impl();

    XrInstance xrInstance() const { return m_instance; }
    WorkQueue& queue() const { return m_workQueue; }
    OpenXRExtensions* extensions() { return m_extensions.get(); }

private:
    XrInstance m_instance { XR_NULL_HANDLE };
    Ref<WorkQueue> m_workQueue;
    std::unique_ptr<OpenXRExtensions> m_extensions;
};

Instance::Impl::Impl()
    : m_workQueue(WorkQueue::create("OpenXR queue"_s))
{
    m_workQueue->dispatch([this]() {
        LOG(XR, "OpenXR: initializing\n");

        m_extensions = OpenXRExtensions::create();
        if (!m_extensions)
            return;

        static constexpr auto s_applicationName = "WebXR (WebKit)"_s;
        static const uint32_t s_applicationVersion = 1;

        const char* const enabledExtensions[] = {
            XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
            XR_MNDX_EGL_ENABLE_EXTENSION_NAME
        };

        auto createInfo = createStructure<XrInstanceCreateInfo, XR_TYPE_INSTANCE_CREATE_INFO>();
        createInfo.createFlags = 0;
        memcpySpan(std::span { createInfo.applicationInfo.applicationName }, s_applicationName.unsafeSpanIncludingNullTerminator());
        createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        createInfo.applicationInfo.applicationVersion = s_applicationVersion;
        createInfo.enabledApiLayerCount = 0;
        createInfo.enabledExtensionCount = std::size(enabledExtensions);
        createInfo.enabledExtensionNames = enabledExtensions;

        XrResult result = xrCreateInstance(&createInfo, &m_instance);
        RETURN_IF_FAILED(result, "xrCreateInstance()", m_instance);
        if (m_instance == XR_NULL_HANDLE)
            return;

        m_extensions->loadMethods(m_instance);

        LOG(XR, "xrCreateInstance(): using instance %p\n", m_instance);
    });
}

Instance::Impl::~Impl()
{
    m_workQueue->dispatch([instance = m_instance] {
        if (instance != XR_NULL_HANDLE)
            xrDestroyInstance(instance);
    });
}

Instance& Instance::singleton()
{
    static LazyNeverDestroyed<Instance> s_instance;
    static std::once_flag s_onceFlag;
    std::call_once(s_onceFlag,
        [&] {
            s_instance.construct();
        });
    return s_instance.get();
}

Instance::Instance()
    : m_impl(makeUniqueRef<Impl>())
{
}

void Instance::enumerateImmersiveXRDevices(CompletionHandler<void(const DeviceList& devices)>&& callback)
{
    m_impl->queue().dispatch([this, callback = WTFMove(callback)]() mutable {
        auto callbackOnExit = makeScopeExit([&]() {
            callOnMainThread([callback = WTFMove(callback)]() mutable {
                callback({ });
            });
        });

        if (m_impl->xrInstance() == XR_NULL_HANDLE || !m_impl->extensions()) {
            LOG(XR, "%s Unable to enumerate XR devices. No XrInstance present\n", __FUNCTION__);
            return;
        }

        auto systemGetInfo = createStructure<XrSystemGetInfo, XR_TYPE_SYSTEM_GET_INFO>();
        systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

        XrSystemId systemId;
        XrResult result = xrGetSystem(m_impl->xrInstance(), &systemGetInfo, &systemId);
        RETURN_IF_FAILED(result, "xrGetSystem", m_impl->xrInstance());

        callbackOnExit.release();

        callOnMainThread([this, callback = WTFMove(callback), systemId]() mutable {
            m_immersiveXRDevices = DeviceList::from(OpenXRDevice::create(m_impl->xrInstance(), systemId, m_impl->queue(), *m_impl->extensions(), [this, callback = WTFMove(callback)]() mutable {
                ASSERT(isMainThread());
                callback(m_immersiveXRDevices);
            }));
        });
    });
}


} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
