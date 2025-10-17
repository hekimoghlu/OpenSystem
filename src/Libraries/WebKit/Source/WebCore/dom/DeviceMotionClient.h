/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#include "DeviceClient.h"
#include <wtf/Noncopyable.h>

namespace WebCore {

class DeviceMotionController;
class DeviceMotionData;
class Page;

class DeviceMotionClient : public DeviceClient {
    WTF_MAKE_TZONE_ALLOCATED(DeviceMotionClient);
    WTF_MAKE_NONCOPYABLE(DeviceMotionClient);
public:
    DeviceMotionClient() = default;
    virtual ~DeviceMotionClient() = default;
    virtual void setController(DeviceMotionController*) = 0;
    virtual DeviceMotionData* lastMotion() const = 0;
    virtual void deviceMotionControllerDestroyed() = 0;

    bool isDeviceMotionClient() const override { return true; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::DeviceMotionClient)
static bool isType(const WebCore::DeviceClient& DeviceClient) { return DeviceClient.isDeviceMotionClient(); }
SPECIALIZE_TYPE_TRAITS_END()

