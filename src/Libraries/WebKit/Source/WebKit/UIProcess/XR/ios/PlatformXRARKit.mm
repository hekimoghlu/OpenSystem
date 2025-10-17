/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#import "PlatformXRARKit.h"

#if ENABLE(WEBXR) && USE(ARKITXR_IOS)

#import "APIUIClient.h"
#import "WKARPresentationSession.h"
#import "WebPageProxy.h"

#import <Metal/MTLEvent_Private.h>
#import <Metal/MTLTexture_Private.h>
#import <WebCore/PlatformXRPose.h>
#import <wtf/TZoneMallocInlines.h>

#import "ARKitSoftLink.h"

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ARKitCoordinator);

struct ARKitCoordinator::RenderState {
    RetainPtr<id<WKARPresentationSession>> presentationSession;
    PlatformXR::Device::RequestFrameCallback onFrameUpdate;
    BinarySemaphore presentFrame;
    std::atomic<bool> terminateRequested;
};

static std::tuple<MachSendRight, bool> makeMachSendRight(id<MTLTexture> texture)
{
    RetainPtr<MTLSharedTextureHandle> sharedTextureHandle = adoptNS([texture newSharedTextureHandle]);
    if (sharedTextureHandle)
        return { MachSendRight::adopt([sharedTextureHandle.get() createMachPort]), true };
    auto surface = WebCore::IOSurface::createFromSurface(texture.iosurface, WebCore::DestinationColorSpace::SRGB());
    if (!surface)
        return { MachSendRight(), false };
    return { surface->createSendRight(), false };
}

ARKitCoordinator::ARKitCoordinator()
{
    ASSERT(RunLoop::isMain());
}

void ARKitCoordinator::getPrimaryDeviceInfo(WebPageProxy&, DeviceInfoCallback&& callback)
{
    RELEASE_LOG(XR, "ARKitCoordinator::getPrimaryDeviceInfo");
    ASSERT(RunLoop::isMain());

    if (![getARWorldTrackingConfigurationClass() isSupported]) {
        RELEASE_LOG_ERROR(XR, "ARKitCoordinator: WorldTrackingConfiguration is not supported");
        return callback(std::nullopt);
    }

    NSArray<ARVideoFormat *> *supportedVideoFormats = [getARWorldTrackingConfigurationClass() supportedVideoFormats];
    if (![supportedVideoFormats count])
        return callback(std::nullopt);

    CGSize recommendedResolution = supportedVideoFormats[0].imageResolution;

    XRDeviceInfo deviceInfo;
    deviceInfo.identifier = m_deviceIdentifier;
    deviceInfo.supportsOrientationTracking = true;
    deviceInfo.supportsStereoRendering = false;
    deviceInfo.recommendedResolution = WebCore::IntSize(recommendedResolution.width, recommendedResolution.height);
    deviceInfo.arFeatures.append(PlatformXR::SessionFeature::ReferenceSpaceTypeUnbounded);
    deviceInfo.arFeatures.append(PlatformXR::SessionFeature::ReferenceSpaceTypeLocalFloor);
    deviceInfo.arFeatures.append(PlatformXR::SessionFeature::ReferenceSpaceTypeLocal);
    deviceInfo.arFeatures.append(PlatformXR::SessionFeature::ReferenceSpaceTypeViewer);

    callback(WTFMove(deviceInfo));
}

void ARKitCoordinator::requestPermissionOnSessionFeatures(WebPageProxy& page, const WebCore::SecurityOriginData& securityOriginData, PlatformXR::SessionMode mode, const PlatformXR::Device::FeatureList& granted, const PlatformXR::Device::FeatureList& consentRequired, const PlatformXR::Device::FeatureList& consentOptional, const PlatformXR::Device::FeatureList& requiredFeaturesRequested, const PlatformXR::Device::FeatureList& optionalFeaturesRequested, FeatureListCallback&& callback)
{
    RELEASE_LOG(XR, "ARKitCoordinator::requestPermissionOnSessionFeatures");
    if (mode == PlatformXR::SessionMode::Inline) {
        callback(granted);
        return;
    }

    page.uiClient().requestPermissionOnXRSessionFeatures(page, securityOriginData, mode, granted, consentRequired, consentOptional, requiredFeaturesRequested, optionalFeaturesRequested, [callback = WTFMove(callback)](std::optional<Vector<PlatformXR::SessionFeature>> userGranted) mutable {
        callback(WTFMove(userGranted));
    });
}

void ARKitCoordinator::startSession(WebPageProxy& page, WeakPtr<SessionEventClient>&& sessionEventClient, const WebCore::SecurityOriginData&, PlatformXR::SessionMode mode, const PlatformXR::Device::FeatureList&)
{
    RELEASE_LOG(XR, "ARKitCoordinator::startSession");
    ASSERT(RunLoop::isMain());
    ASSERT(mode == PlatformXR::SessionMode::ImmersiveAr);

    WTF::switchOn(m_state,
        [&](Idle&) {
            createSessionIfNeeded();

            // FIXME: When in element fullscreen, UIClient::presentingViewController() may not return the
            // WKFullScreenViewController even though that is the presenting view controller of the WKWebView.
            // We should call PageClientImpl::presentingViewController() instead.
            auto* presentingViewController = page.uiClient().presentingViewController();
            if (!presentingViewController) {
                RELEASE_LOG_ERROR(XR, "ARKitCoordinator: failed to obtain presenting ViewController from page.");
                return;
            }

            auto presentationSessionDesc = adoptNS([WKARPresentationSessionDescriptor new]);
            [presentationSessionDesc setPresentingViewController:presentingViewController];

            auto presentationSession = adoptNS(createPresentationSession(m_session.get(), presentationSessionDesc.get()));

            auto renderState = Box<RenderState>::create();
            renderState->presentationSession = WTFMove(presentationSession);
            renderState->terminateRequested = false;

            m_state = Active {
                .sessionEventClient = WTFMove(sessionEventClient),
                .pageIdentifier = page.webPageIDInMainFrameProcess(),
                .renderState = renderState,
                .renderThread = Thread::create("ARKitCoordinator session renderer"_s, [this, renderState] { renderLoop(renderState); }),
            };
        },
        [&](Active&) {
            RELEASE_LOG_ERROR(XR, "ARKitCoordinator: an existing immersive session is active");
            if (RefPtr protectedSessionEventClient = sessionEventClient.get())
                protectedSessionEventClient->sessionDidEnd(m_deviceIdentifier);
        });
}

void ARKitCoordinator::endSessionIfExists(std::optional<WebCore::PageIdentifier> pageIdentifier)
{
    RELEASE_LOG(XR, "ARKitCoordinator::endSessionIfExists");
    ASSERT(RunLoop::isMain());

    WTF::switchOn(m_state,
        [&](Idle&) { },
        [&](Active& active) {
            if (pageIdentifier && active.pageIdentifier != *pageIdentifier) {
                RELEASE_LOG(XR, "ARKitCoordinator: trying to end an immersive session owned by another page");
                return;
            }

            if (active.renderState->terminateRequested)
                return;

            active.renderState->terminateRequested = true;
            active.renderState->presentFrame.signal();
            active.renderThread->waitForCompletion();

            if (active.renderState->onFrameUpdate)
                active.renderState->onFrameUpdate({ });

            auto& sessionEventClient = active.sessionEventClient;
            if (sessionEventClient) {
                RELEASE_LOG(XR, "... immersive session end sent");
                sessionEventClient->sessionDidEnd(m_deviceIdentifier);
            }

            m_state = Idle { };
        });
}


void ARKitCoordinator::endSessionIfExists(WebPageProxy& page)
{
    endSessionIfExists(page.webPageIDInMainFrameProcess());
}

void ARKitCoordinator::scheduleAnimationFrame(WebPageProxy& page, std::optional<PlatformXR::RequestData>&&, PlatformXR::Device::RequestFrameCallback&& onFrameUpdateCallback)
{
    RELEASE_LOG(XR, "ARKitCoordinator::scheduleAnimationFrame");
    WTF::switchOn(m_state,
        [&](Idle&) {
            RELEASE_LOG(XR, "ARKitCoordinator: trying to schedule frame update for an inactive session");
            onFrameUpdateCallback({ });
        },
        [&](Active& active) {
            if (active.pageIdentifier != page.webPageIDInMainFrameProcess()) {
                RELEASE_LOG(XR, "ARKitCoordinator: trying to schedule frame update for session owned by another page");
                return;
            }

            if (active.renderState->terminateRequested) {
                RELEASE_LOG(XR, "ARKitCoordinator: trying to schedule frame for terminating session");
                onFrameUpdateCallback({ });
            }

            active.renderState->onFrameUpdate = WTFMove(onFrameUpdateCallback);
        });
}

void ARKitCoordinator::submitFrame(WebPageProxy& page)
{
    RELEASE_LOG(XR, "ARKitCoordinator::submitFrame");
    ASSERT(RunLoop::isMain());
    WTF::switchOn(m_state,
        [&](Idle&) {
            RELEASE_LOG(XR, "ARKitCoordinator: trying to submit frame update for an inactive session");
        },
        [&](Active& active) {
            if (active.pageIdentifier != page.webPageIDInMainFrameProcess()) {
                RELEASE_LOG(XR, "ARKitCoordinator: trying to submit frame update for session owned by another page");
                return;
            }

            if (WTF::atomicLoad(&active.renderState->terminateRequested)) {
                RELEASE_LOG(XR, "ARKitCoordinator: trying to submit frame update for a terminating session");
                return;
            }

            // FIXME: rdar://118492973 (Re-enable MTLSharedEvent completion sync)
            // Replace frame presentation to depend on
            // active.presentationSession.completionEvent
            active.renderState->presentFrame.signal();
        });
}

void ARKitCoordinator::createSessionIfNeeded()
{
    ASSERT(RunLoop::isMain());
    RELEASE_LOG(XR, "ARKitCoordinator::createSessionIfNeeded");

    if (m_session)
        return;

    m_session = adoptNS([WebKit::allocARSessionInstance() init]);
}

void ARKitCoordinator::renderLoop(Box<RenderState> active)
{
    for (;;) {
        if (active->terminateRequested)
            break;

        if ([active->presentationSession isSessionEndRequested]) {
            callOnMainRunLoop([this]() {
                endSessionIfExists(std::nullopt);
            });
            break;
        }

        if (!active->onFrameUpdate)
            continue;

        @autoreleasepool {
            id<WKARPresentationSession> presentationSession = active->presentationSession.get();
            [presentationSession startFrame];

            ARFrame* frame = presentationSession.currentFrame;
            ARCamera* camera = frame.camera;

            PlatformXR::FrameData frameData = { };
            // FIXME: Use ARSession state to calculate correct values.
            frameData.isTrackingValid = true;
            frameData.isPositionValid = true;
            frameData.predictedDisplayTime = frame.timestamp;
            frameData.origin = PlatformXRPose(camera.transform).pose();

            frameData.inputSources = [presentationSession collectInputSources];

            // Only one view
            frameData.views.append({
                .offset = { },
                .projection = {
                    PlatformXRPose(frame.camera.projectionMatrix).toColumnMajorFloatArray()
                },
            });
            id<MTLTexture> colorTexture = presentationSession.colorTexture;
            auto colorTextureSendRight = makeMachSendRight(colorTexture);
            auto renderingFrameIndex = presentationSession.renderingFrameIndex;
            // FIXME: Send this event once at setup time, not every frame.
            id<MTLSharedEvent> completionEvent = presentationSession.completionEvent;
            RetainPtr<MTLSharedEventHandle> completionHandle = adoptNS([completionEvent newSharedEventHandle]);
            auto completionPort = MachSendRight::create([completionHandle.get() eventPort]);

            // FIXME: rdar://77858090 (Need to transmit color space information)
            auto layerData = makeUniqueRef<PlatformXR::FrameData::LayerData>(PlatformXR::FrameData::LayerData {
                .framebufferSize = IntSize(colorTexture.width, colorTexture.height),
                .textureData = PlatformXR::FrameData::ExternalTextureData {
                    .colorTexture = WTFMove(colorTextureSendRight),
                    .completionSyncEvent = { MachSendRight(completionPort), renderingFrameIndex }
                },
            });
            frameData.layers.set(defaultLayerHandle(), WTFMove(layerData));
            frameData.shouldRender = true;

            callOnMainRunLoop([callback = WTFMove(active->onFrameUpdate), frameData = WTFMove(frameData)]() mutable {
                callback(WTFMove(frameData));
            });

            active->presentFrame.wait();

            [presentationSession present];
        }
    }

    RELEASE_LOG(XR, "ARKitCoordinator::renderLoop exiting...");
}

} // namespace WebKit

#endif // ENABLE(WEBXR) && USE(ARKITXR_IOS)
