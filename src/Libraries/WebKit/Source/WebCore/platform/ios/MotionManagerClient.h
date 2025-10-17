/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class MotionManagerClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MotionManagerClient> : std::true_type { };
}

namespace WebCore {

class MotionManagerClient : public CanMakeWeakPtr<MotionManagerClient> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MotionManagerClient);
public:
    virtual ~MotionManagerClient() { };

    virtual void orientationChanged(double, double, double, double, double) { }
    virtual void motionChanged(double, double, double, double, double, double, std::optional<double>, std::optional<double>, std::optional<double>) { }
};

};

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
