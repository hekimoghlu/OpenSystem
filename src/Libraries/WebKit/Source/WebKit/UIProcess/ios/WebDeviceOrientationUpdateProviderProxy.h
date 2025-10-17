/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

#include "MessageReceiver.h"
#include <WebCore/MotionManagerClient.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebPageProxy;
struct SharedPreferencesForWebProcess;

class WebDeviceOrientationUpdateProviderProxy : public WebCore::MotionManagerClient, private IPC::MessageReceiver, public RefCounted<WebDeviceOrientationUpdateProviderProxy> {
    WTF_MAKE_TZONE_ALLOCATED(WebDeviceOrientationUpdateProviderProxy);
public:
    static Ref<WebDeviceOrientationUpdateProviderProxy> create(WebPageProxy&);
    ~WebDeviceOrientationUpdateProviderProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void startUpdatingDeviceOrientation();
    void stopUpdatingDeviceOrientation();

    void startUpdatingDeviceMotion();
    void stopUpdatingDeviceMotion();

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    explicit WebDeviceOrientationUpdateProviderProxy(WebPageProxy&);

    // WebCore::WebCoreMotionManagerClient
    void orientationChanged(double, double, double, double, double) final;
    void motionChanged(double, double, double, double, double, double, std::optional<double>, std::optional<double>, std::optional<double>) final;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    WeakPtr<WebPageProxy> m_page;
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
