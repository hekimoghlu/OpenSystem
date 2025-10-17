/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#import "config.h"
#import "WebDeviceOrientationUpdateProviderProxy.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

#import "MessageSenderInlines.h"
#import "WebDeviceOrientationUpdateProviderMessages.h"
#import "WebDeviceOrientationUpdateProviderProxyMessages.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import <WebCore/WebCoreMotionManager.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebDeviceOrientationUpdateProviderProxy);

Ref<WebDeviceOrientationUpdateProviderProxy> WebDeviceOrientationUpdateProviderProxy::create(WebPageProxy& page)
{
    return adoptRef(*new WebDeviceOrientationUpdateProviderProxy(page));
}

WebDeviceOrientationUpdateProviderProxy::WebDeviceOrientationUpdateProviderProxy(WebPageProxy& page)
    : m_page(page)
{
    page.protectedLegacyMainFrameProcess()->addMessageReceiver(Messages::WebDeviceOrientationUpdateProviderProxy::messageReceiverName(), page.webPageIDInMainFrameProcess(), *this);
}

WebDeviceOrientationUpdateProviderProxy::~WebDeviceOrientationUpdateProviderProxy()
{
    if (RefPtr page = m_page.get())
        page->protectedLegacyMainFrameProcess()->removeMessageReceiver(Messages::WebDeviceOrientationUpdateProviderProxy::messageReceiverName(), page->webPageIDInMainFrameProcess());
}

void WebDeviceOrientationUpdateProviderProxy::startUpdatingDeviceOrientation()
{
    [[WebCoreMotionManager sharedManager] addOrientationClient:this];
}

void WebDeviceOrientationUpdateProviderProxy::stopUpdatingDeviceOrientation()
{
    [[WebCoreMotionManager sharedManager] removeOrientationClient:this];
}

void WebDeviceOrientationUpdateProviderProxy::startUpdatingDeviceMotion()
{
    [[WebCoreMotionManager sharedManager] addMotionClient:this];
}

void WebDeviceOrientationUpdateProviderProxy::stopUpdatingDeviceMotion()
{
    [[WebCoreMotionManager sharedManager] removeMotionClient:this];
}

void WebDeviceOrientationUpdateProviderProxy::orientationChanged(double alpha, double beta, double gamma, double compassHeading, double compassAccuracy)
{
    if (RefPtr page = m_page.get())
        page->protectedLegacyMainFrameProcess()->send(Messages::WebDeviceOrientationUpdateProvider::DeviceOrientationChanged(alpha, beta, gamma, compassHeading, compassAccuracy), m_page->webPageIDInMainFrameProcess());
}

void WebDeviceOrientationUpdateProviderProxy::motionChanged(double xAcceleration, double yAcceleration, double zAcceleration, double xAccelerationIncludingGravity, double yAccelerationIncludingGravity, double zAccelerationIncludingGravity, std::optional<double> xRotationRate, std::optional<double> yRotationRate, std::optional<double> zRotationRate)
{
    if (RefPtr page = m_page.get())
        page->protectedLegacyMainFrameProcess()->send(Messages::WebDeviceOrientationUpdateProvider::DeviceMotionChanged(xAcceleration, yAcceleration, zAcceleration, xAccelerationIncludingGravity, yAccelerationIncludingGravity, zAccelerationIncludingGravity, xRotationRate, yRotationRate, zRotationRate), m_page->webPageIDInMainFrameProcess());
}

std::optional<SharedPreferencesForWebProcess> WebDeviceOrientationUpdateProviderProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr page = m_page.get())
        return m_page->legacyMainFrameProcess().sharedPreferencesForWebProcess();
    return std::nullopt;
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
