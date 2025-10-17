/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
#include "WebGPUCreateImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUAdapterImpl.h"
#include "WebGPUDowncastConvertToBackingContext.h"
#include "WebGPUImpl.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/BlockPtr.h>

#if PLATFORM(COCOA)
#include <wtf/darwin/WeakLinking.h>

WTF_WEAK_LINK_FORCE_IMPORT(wgpuCreateInstance);
#endif

namespace WebCore::WebGPU {

RefPtr<GPU> create(ScheduleWorkFunction&& scheduleWorkFunction, const WebCore::ProcessIdentity* webProcessIdentity)
{
#if !HAVE(TASK_IDENTITY_TOKEN)
    UNUSED_PARAM(webProcessIdentity);
#endif
    auto scheduleWorkBlock = makeBlockPtr([scheduleWorkFunction = WTFMove(scheduleWorkFunction)](WGPUWorkItem workItem)
    {
        scheduleWorkFunction(Function<void()>(makeBlockPtr(WTFMove(workItem))));
    });
    WGPUInstanceCocoaDescriptor cocoaDescriptor {
        {
            nullptr,
            static_cast<WGPUSType>(WGPUSTypeExtended_InstanceCocoaDescriptor),
        },
        scheduleWorkBlock.get(),
#if HAVE(TASK_IDENTITY_TOKEN)
        webProcessIdentity ? &webProcessIdentity->taskId() : nullptr,
#else
        nullptr,
#endif
    };
    WGPUInstanceDescriptor descriptor = { &cocoaDescriptor.chain };

    if (!&wgpuCreateInstance)
        return nullptr;
    auto instance = adoptWebGPU(wgpuCreateInstance(&descriptor));
    if (!instance)
        return nullptr;
    auto convertToBackingContext = DowncastConvertToBackingContext::create();
    return GPUImpl::create(WTFMove(instance), convertToBackingContext);
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
