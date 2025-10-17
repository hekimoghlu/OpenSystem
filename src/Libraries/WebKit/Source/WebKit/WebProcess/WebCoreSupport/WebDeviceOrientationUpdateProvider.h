/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#include "MessageReceiver.h"
#include "WebPage.h"

#include <WebCore/DeviceOrientationUpdateProvider.h>

#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

namespace WebKit {

class WebDeviceOrientationUpdateProvider final : public WebCore::DeviceOrientationUpdateProvider, private IPC::MessageReceiver {
public:
    static Ref<WebDeviceOrientationUpdateProvider> create(WebPage& page) { return adoptRef(*new WebDeviceOrientationUpdateProvider(page));}

    void ref() const final { WebCore::DeviceOrientationUpdateProvider::ref(); }
    void deref() const final { WebCore::DeviceOrientationUpdateProvider::deref(); }

private:
    WebDeviceOrientationUpdateProvider(WebPage&);
    ~WebDeviceOrientationUpdateProvider();

    // WebCore::DeviceOrientationUpdateProvider
    void startUpdatingDeviceOrientation(WebCore::MotionManagerClient&) final;
    void stopUpdatingDeviceOrientation(WebCore::MotionManagerClient&) final;
    void startUpdatingDeviceMotion(WebCore::MotionManagerClient&) final;
    void stopUpdatingDeviceMotion(WebCore::MotionManagerClient&) final;
    void deviceOrientationChanged(double, double, double, double, double) final;
    void deviceMotionChanged(double, double, double, double, double, double, std::optional<double>, std::optional<double>, std::optional<double>) final;

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    WeakPtr<WebPage> m_page;
    WebCore::PageIdentifier m_pageIdentifier;
    WeakHashSet<WebCore::MotionManagerClient> m_deviceOrientationClients;
    WeakHashSet<WebCore::MotionManagerClient> m_deviceMotionClients;
};

}

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
