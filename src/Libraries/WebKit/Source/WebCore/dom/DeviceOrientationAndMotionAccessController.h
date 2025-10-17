/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

#if ENABLE(DEVICE_ORIENTATION)

#include "DeviceOrientationOrMotionPermissionState.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "SecurityOriginData.h"
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class DeviceOrientationAndMotionAccessController;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::DeviceOrientationAndMotionAccessController> : std::true_type { };
}

namespace WebCore {

class Document;
class Page;

class DeviceOrientationAndMotionAccessController : public CanMakeWeakPtr<DeviceOrientationAndMotionAccessController> {
    WTF_MAKE_TZONE_ALLOCATED(DeviceOrientationAndMotionAccessController);
public:
    explicit DeviceOrientationAndMotionAccessController(Document& topDocument);

    DeviceOrientationOrMotionPermissionState accessState(const Document&) const;
    void shouldAllowAccess(const Document&, Function<void(DeviceOrientationOrMotionPermissionState)>&&);

private:
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_topDocument;
    UncheckedKeyHashMap<SecurityOriginData, DeviceOrientationOrMotionPermissionState> m_accessStatePerOrigin;
};

} // namespace WebCore

#endif // ENABLE(DEVICE_ORIENTATION)
