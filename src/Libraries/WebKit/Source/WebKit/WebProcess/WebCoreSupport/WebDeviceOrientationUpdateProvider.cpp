/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#include "config.h"
#include "WebDeviceOrientationUpdateProvider.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

#include "MessageSenderInlines.h"
#include "WebDeviceOrientationUpdateProviderMessages.h"
#include "WebDeviceOrientationUpdateProviderProxyMessages.h"
#include "WebPage.h"
#include "WebProcess.h"

#include <WebCore/MotionManagerClient.h>

namespace WebKit {

WebDeviceOrientationUpdateProvider::WebDeviceOrientationUpdateProvider(WebPage& page)
    : m_page(page)
    , m_pageIdentifier(page.identifier())
{
    WebProcess::singleton().addMessageReceiver(Messages::WebDeviceOrientationUpdateProvider::messageReceiverName(), page.identifier(), *this);
}

WebDeviceOrientationUpdateProvider::~WebDeviceOrientationUpdateProvider()
{
    WebProcess::singleton().removeMessageReceiver(Messages::WebDeviceOrientationUpdateProvider::messageReceiverName(), m_pageIdentifier);
}

void WebDeviceOrientationUpdateProvider::startUpdatingDeviceOrientation(WebCore::MotionManagerClient& client)
{
    if (m_deviceOrientationClients.isEmptyIgnoringNullReferences() && m_page)
        m_page->send(Messages::WebDeviceOrientationUpdateProviderProxy::StartUpdatingDeviceOrientation());
    m_deviceOrientationClients.add(client);
}

void WebDeviceOrientationUpdateProvider::stopUpdatingDeviceOrientation(WebCore::MotionManagerClient& client)
{
    if (m_deviceOrientationClients.isEmptyIgnoringNullReferences())
        return;
    m_deviceOrientationClients.remove(client);
    if (m_deviceOrientationClients.isEmptyIgnoringNullReferences() && m_page)
        m_page->send(Messages::WebDeviceOrientationUpdateProviderProxy::StopUpdatingDeviceOrientation());
}

void WebDeviceOrientationUpdateProvider::startUpdatingDeviceMotion(WebCore::MotionManagerClient& client)
{
    if (m_deviceMotionClients.isEmptyIgnoringNullReferences() && m_page)
        m_page->send(Messages::WebDeviceOrientationUpdateProviderProxy::StartUpdatingDeviceMotion());
    m_deviceMotionClients.add(client);
}

void WebDeviceOrientationUpdateProvider::stopUpdatingDeviceMotion(WebCore::MotionManagerClient& client)
{
    if (m_deviceMotionClients.isEmptyIgnoringNullReferences())
        return;
    m_deviceMotionClients.remove(client);
    if (m_deviceMotionClients.isEmptyIgnoringNullReferences() && m_page)
        m_page->send(Messages::WebDeviceOrientationUpdateProviderProxy::StopUpdatingDeviceMotion());
}

void WebDeviceOrientationUpdateProvider::deviceOrientationChanged(double alpha, double beta, double gamma, double compassHeading, double compassAccuracy)
{
    for (auto& client : copyToVectorOf<WeakPtr<WebCore::MotionManagerClient>>(m_deviceOrientationClients)) {
        if (client)
            client->orientationChanged(alpha, beta, gamma, compassHeading, compassAccuracy);
    }
}

void WebDeviceOrientationUpdateProvider::deviceMotionChanged(double xAcceleration, double yAcceleration, double zAcceleration, double xAccelerationIncludingGravity, double yAccelerationIncludingGravity, double zAccelerationIncludingGravity, std::optional<double> xRotationRate, std::optional<double> yRotationRate, std::optional<double> zRotationRate)
{
    for (auto& client : copyToVectorOf<WeakPtr<WebCore::MotionManagerClient>>(m_deviceMotionClients)) {
        if (client)
            client->motionChanged(xAcceleration, yAcceleration, zAcceleration, xAccelerationIncludingGravity, yAccelerationIncludingGravity,  zAccelerationIncludingGravity, xRotationRate, yRotationRate, zRotationRate);
    }
}

}

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
