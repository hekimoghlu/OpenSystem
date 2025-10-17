/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

#include "DeviceController.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DeviceMotionClient;
class DeviceMotionData;

class DeviceMotionController : public DeviceController {
    WTF_MAKE_TZONE_ALLOCATED(DeviceMotionController);
    WTF_MAKE_NONCOPYABLE(DeviceMotionController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DeviceMotionController);
public:
    explicit DeviceMotionController(DeviceMotionClient&);
    virtual ~DeviceMotionController() = default;

#if PLATFORM(IOS_FAMILY)
    // FIXME: We should look to reconcile the iOS and OpenSource differences with this class
    // so that we can either remove these methods or remove the PLATFORM(IOS_FAMILY)-guard.
    void suspendUpdates();
    void resumeUpdates();
#endif

    void didChangeDeviceMotion(DeviceMotionData*);
    DeviceMotionClient& deviceMotionClient();

    bool hasLastData() override;
    RefPtr<Event> getLastEvent() override;

    static ASCIILiteral supplementName();
    static DeviceMotionController* from(Page*);
    static bool isActiveAt(Page*);
};

} // namespace WebCore
