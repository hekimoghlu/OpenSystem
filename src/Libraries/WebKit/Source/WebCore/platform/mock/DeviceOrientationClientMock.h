/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#ifndef DeviceOrientationClientMock_h
#define DeviceOrientationClientMock_h

#include "DeviceOrientationClient.h"
#include "DeviceOrientationData.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DeviceOrientationController;

// A mock implementation of DeviceOrientationClient used to test the feature in
// DumpRenderTree. Embedders should should configure the Page object to use this
// client when running DumpRenderTree.
class DeviceOrientationClientMock final : public DeviceOrientationClient, public CanMakeCheckedPtr<DeviceOrientationClientMock> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(DeviceOrientationClientMock, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DeviceOrientationClientMock);
public:
    WEBCORE_EXPORT DeviceOrientationClientMock();

    // DeviceOrientationClient
    WEBCORE_EXPORT void setController(DeviceOrientationController*) override;
    WEBCORE_EXPORT void startUpdating() override;
    WEBCORE_EXPORT void stopUpdating() override;
    DeviceOrientationData* lastOrientation() const override { return m_orientation.get(); }
    void deviceOrientationControllerDestroyed() override { }

    WEBCORE_EXPORT void setOrientation(RefPtr<DeviceOrientationData>&&);

private:
    void timerFired();

    RefPtr<DeviceOrientationData> m_orientation;
    DeviceOrientationController* m_controller;
    Timer m_timer;
    bool m_isUpdating;
};

} // namespace WebCore

#endif // DeviceOrientationClientMock_h
