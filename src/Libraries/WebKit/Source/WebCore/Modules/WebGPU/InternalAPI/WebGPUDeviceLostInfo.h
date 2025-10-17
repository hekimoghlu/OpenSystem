/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

#include "WebGPUDeviceLostReason.h"
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class DeviceLostInfo final : public RefCounted<DeviceLostInfo> {
public:
    static Ref<DeviceLostInfo> create(DeviceLostReason reason, String&& message)
    {
        return adoptRef(*new DeviceLostInfo(reason, WTFMove(message)));
    }

    DeviceLostReason reason() const { return m_reason; }
    const String& message() const { return m_message; }

protected:
    DeviceLostInfo(DeviceLostReason reason, String&& message)
        : m_reason(reason)
        , m_message(WTFMove(message))
    {
    }

private:
    DeviceLostInfo(const DeviceLostInfo&) = delete;
    DeviceLostInfo(DeviceLostInfo&&) = delete;
    DeviceLostInfo& operator=(const DeviceLostInfo&) = delete;
    DeviceLostInfo& operator=(DeviceLostInfo&&) = delete;

    DeviceLostReason m_reason;
    String m_message;
};

} // namespace WebCore::WebGPU
