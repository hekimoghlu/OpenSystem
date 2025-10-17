/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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

class DeviceOrientationClient;
class DeviceOrientationData;
class Page;

class DeviceOrientationController final : public DeviceController {
    WTF_MAKE_TZONE_ALLOCATED(DeviceOrientationController);
    WTF_MAKE_NONCOPYABLE(DeviceOrientationController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DeviceOrientationController);
public:
    explicit DeviceOrientationController(DeviceOrientationClient&);
    virtual ~DeviceOrientationController() = default;

    void didChangeDeviceOrientation(DeviceOrientationData*);
    DeviceOrientationClient& deviceOrientationClient();

#if PLATFORM(IOS_FAMILY)
    // FIXME: We should look to reconcile the iOS and OpenSource differences with this class
    // so that we can either remove these methods or remove the PLATFORM(IOS_FAMILY)-guard.
    void suspendUpdates();
    void resumeUpdates();
#else
    bool hasLastData() override;
    RefPtr<Event> getLastEvent() override;
#endif

    static ASCIILiteral supplementName();
    static DeviceOrientationController* from(Page*);
    static bool isActiveAt(Page*);
};

} // namespace WebCore
