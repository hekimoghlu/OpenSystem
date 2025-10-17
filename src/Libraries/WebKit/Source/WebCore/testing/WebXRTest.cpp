/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#include "WebXRTest.h"

#if ENABLE(WEBXR)

#include "JSDOMPromiseDeferred.h"
#include "JSWebFakeXRDevice.h"
#include "JSXRReferenceSpaceType.h"
#include "PlatformXR.h"
#include "UserGestureIndicator.h"
#include "WebXRSystem.h"
#include "XRSessionMode.h"

namespace WebCore {

WebXRTest::WebXRTest(WeakPtr<WebXRSystem, WeakPtrImplWithEventTargetData>&& system)
    : m_context(WTFMove(system))
{
}

WebXRTest::~WebXRTest() = default;

static PlatformXR::Device::FeatureList parseFeatures(const Vector<JSC::JSValue>& featureList, ScriptExecutionContext& context)
{
    PlatformXR::Device::FeatureList features;
    if (auto* globalObject = context.globalObject()) {
        for (auto& feature : featureList) {
            auto featureString = feature.toWTFString(globalObject);
            if (auto sessionFeature = PlatformXR::parseSessionFeatureDescriptor(featureString))
                features.append(*sessionFeature);
        }
    }
    return features;
}

void WebXRTest::simulateDeviceConnection(ScriptExecutionContext& context, const FakeXRDeviceInit& init, WebFakeXRDevicePromise&& promise)
{
    // https://immersive-web.github.io/webxr-test-api/#dom-xrtest-simulatedeviceconnection
    context.postTask([this, protectedThis = Ref { *this }, init, promise = WTFMove(promise)] (ScriptExecutionContext& context) mutable {
        auto device = WebFakeXRDevice::create();
        auto& simulatedDevice = device->simulatedXRDevice();

        device->setViews(init.views);

        PlatformXR::Device::FeatureList supportedFeatures;
        if (init.supportedFeatures)
            supportedFeatures = parseFeatures(init.supportedFeatures.value(), context);
        PlatformXR::Device::FeatureList enabledFeatures;
        if (init.enabledFeatures)
            enabledFeatures = parseFeatures(init.enabledFeatures.value(), context);

        if (init.boundsCoordinates) {
            if (init.boundsCoordinates->size() < 3) {
                promise.reject(Exception { ExceptionCode::TypeError });
                return;
            }
            simulatedDevice.setNativeBoundsGeometry(init.boundsCoordinates.value());
        }

        if (init.viewerOrigin)
            device->setViewerOrigin(init.viewerOrigin.value());

        if (init.floorOrigin)
            device->setFloorOrigin(init.floorOrigin.value());

        Vector<XRSessionMode> supportedModes;
        if (init.supportedModes) {
            supportedModes = init.supportedModes.value();
            if (supportedModes.isEmpty())
                supportedModes.append(XRSessionMode::Inline);
        } else {
            supportedModes.append(XRSessionMode::Inline);
            if (init.supportsImmersive)
                supportedModes.append(XRSessionMode::ImmersiveVr);
        }

        for (auto& mode : supportedModes) {
            simulatedDevice.setSupportedFeatures(mode, supportedFeatures);
            simulatedDevice.setEnabledFeatures(mode, enabledFeatures);
        }

        m_context->registerSimulatedXRDeviceForTesting(simulatedDevice);

        promise.resolve(device.get());
        m_devices.append(WTFMove(device));
    });
}

void WebXRTest::simulateUserActivation(Document& document, XRSimulateUserActivationFunction& function)
{
    // https://immersive-web.github.io/webxr-test-api/#dom-xrtest-simulateuseractivation
    // Invoke function as if it had transient activation.
    UserGestureIndicator gestureIndicator(IsProcessingUserGesture::Yes, &document);
    function.handleEvent();
}

void WebXRTest::disconnectAllDevices(DOMPromiseDeferred<void>&& promise)
{
    for (auto& device : m_devices)
        m_context->unregisterSimulatedXRDeviceForTesting(device->simulatedXRDevice());
    m_devices.clear();
    promise.resolve();
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
