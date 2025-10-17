/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

#include "WebGPUDeviceDescriptor.h"
#include "WebGPUSupportedFeatures.h"
#include "WebGPUSupportedLimits.h"
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class Device;

class Adapter : public RefCountedAndCanMakeWeakPtr<Adapter> {
public:
    virtual ~Adapter() = default;

    const String& name() const { return m_name; }
    SupportedFeatures& features() const { return m_features; }
    SupportedLimits& limits() const { return m_limits; }
    bool isFallbackAdapter() const { return m_isFallbackAdapter; }
    virtual bool xrCompatible() = 0;

    virtual void requestDevice(const DeviceDescriptor&, CompletionHandler<void(RefPtr<Device>&&)>&&) = 0;

protected:
    Adapter(String&& name, SupportedFeatures& features, SupportedLimits& limits, bool isFallbackAdapter)
        : m_name(WTFMove(name))
        , m_features(features)
        , m_limits(limits)
        , m_isFallbackAdapter(isFallbackAdapter)
    {
    }

private:
    Adapter(const Adapter&) = delete;
    Adapter(Adapter&&) = delete;
    Adapter& operator=(const Adapter&) = delete;
    Adapter& operator=(Adapter&&) = delete;

    String m_name;
    Ref<SupportedFeatures> m_features;
    Ref<SupportedLimits> m_limits;
    bool m_isFallbackAdapter;
};

} // namespace WebCore::WebGPU
