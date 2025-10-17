/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include "WebXRFrame.h"

#if ENABLE(WEBXR)

#include "WebXRBoundedReferenceSpace.h"
#include "WebXRJointPose.h"
#include "WebXRJointSpace.h"
#include "WebXRReferenceSpace.h"
#include "WebXRSession.h"
#include "WebXRViewerPose.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRFrame);

Ref<WebXRFrame> WebXRFrame::create(WebXRSession& session, IsAnimationFrame isAnimationFrame)
{
    return adoptRef(*new WebXRFrame(session, isAnimationFrame));
}

WebXRFrame::WebXRFrame(WebXRSession& session, IsAnimationFrame isAnimationFrame)
    : m_isAnimationFrame(isAnimationFrame == IsAnimationFrame::Yes)
    , m_session(session)
{
}

WebXRFrame::~WebXRFrame() = default;

bool WebXRFrame::isOutsideNativeBoundsOfBoundedReferenceSpace(const WebXRSpace& space, const WebXRSpace&) const
{
    if (!is<WebXRBoundedReferenceSpace>(space))
        return false;

    // FIXME: return true whenever the distance from the bounded geometry of
    // |space| to the native origin of |other| space is greater than 1m
    // (suggested by specs).

    return false;
}

bool WebXRFrame::isLocalReferenceSpace(const WebXRSpace& space) const
{
    auto* referenceSpace = dynamicDowncast<WebXRReferenceSpace>(space);
    if (!referenceSpace)
        return false;

    auto type = referenceSpace->type();
    if (type == XRReferenceSpaceType::Local || type == XRReferenceSpaceType::LocalFloor)
        return true;

    return false;
}

// https://immersive-web.github.io/webxr/#poses-must-be-limited
bool WebXRFrame::mustPosesBeLimited(const WebXRSpace& space, const WebXRSpace& baseSpace) const
{
    if (isOutsideNativeBoundsOfBoundedReferenceSpace(space, baseSpace)
        || isOutsideNativeBoundsOfBoundedReferenceSpace(baseSpace, space))
        return true;

    if (isLocalReferenceSpace(space) || isLocalReferenceSpace(baseSpace)) {
        // FIXME: If the distance between native origins of spaces is greater
        // than 15m (suggested by specs) return true.
    }

    return false;
}

struct WebXRFrame::PopulatedPose {
    TransformationMatrix transform;
    bool emulatedPosition { false };
};

// https://immersive-web.github.io/webxr/#populate-the-pose
ExceptionOr<std::optional<WebXRFrame::PopulatedPose>> WebXRFrame::populatePose(const Document& document, const WebXRSpace& space, const WebXRSpace& baseSpace)
{
    // 1. If frameâ€™s active boolean is false, throw an InvalidStateError and abort these steps.
    if (!m_active)
        return Exception { ExceptionCode::InvalidStateError };

    // 2. Let session be frameâ€™s session object.
    // 3. If spaceâ€™s session does not equal session, throw an InvalidStateError and abort these steps.
    if (space.session() != m_session.ptr())
        return Exception { ExceptionCode::InvalidStateError };

    // 4. If baseSpaceâ€™s session does not equal session, throw an InvalidStateError and abort these steps.
    if (baseSpace.session() != m_session.ptr())
        return Exception { ExceptionCode::InvalidStateError };

    // 5. Check if poses may be reported and, if not, throw a SecurityError and abort these steps.
    if (!m_session->posesCanBeReported(document))
        return Exception { ExceptionCode::SecurityError };

    // 6. Let limit be the result of whether poses must be limited between space and baseSpace.
    // 7. Let transform be poseâ€™s transform.
    // 8. Query the XR device's tracking system for spaceâ€™s pose relative to baseSpace at the frameâ€™s time.

    if (m_isAnimationFrame && !m_session->frameData().isTrackingValid) {
        // FIXME: check if spaceâ€™s pose relative to baseSpace has been determined in the past.
        // Anyway this emulation is usually provided by the system in the pose (e.g. OpenXR)
        // so we shouldn't hit this path in most XRPlatform ports.
        return { std::nullopt };
    }

    auto baseTransform = baseSpace.effectiveOrigin();
    if (!baseTransform)
        return Exception { ExceptionCode::InvalidStateError };

    if (!baseTransform.value().isInvertible())
        return { std::nullopt };

    auto effectiveOrigin = space.effectiveOrigin();
    // A space's effectiveOrigin can be null, such as a joint pose from a hand that has
    // other missing joint poses.
    if (!effectiveOrigin)
        return { std::nullopt };

    auto transform =  *baseTransform.value().inverse() * effectiveOrigin.value();

    auto isPositionEmulated = space.isPositionEmulated();
    if (!isPositionEmulated)
        return Exception { ExceptionCode::InvalidStateError };

    auto baseSpaceIsPositionEmulated = baseSpace.isPositionEmulated();
    if (!baseSpaceIsPositionEmulated)
        return Exception { ExceptionCode::InvalidStateError };

    bool emulatedPosition = isPositionEmulated.value() || baseSpaceIsPositionEmulated.value();

    bool limit = mustPosesBeLimited(space, baseSpace);
    if (limit) {
        // FIXME: apply pose limits logic
        // https://immersive-web.github.io/webxr/#poses-must-be-limited
    }

    return { PopulatedPose { transform, emulatedPosition } };
}

// https://immersive-web.github.io/webxr/#dom-xrframe-getviewerpose
ExceptionOr<RefPtr<WebXRViewerPose>> WebXRFrame::getViewerPose(const Document& document, const WebXRReferenceSpace& referenceSpace)
{
    // 1. Let frame be this.
    // 2. Let session be frameâ€™s session object.
    // 3. If frameâ€™s animationFrame boolean is false, throw an InvalidStateError and abort these steps.
    if (!m_isAnimationFrame)
        return Exception { ExceptionCode::InvalidStateError };

    // 4. Let pose be a new XRViewerPose object in the relevant realm of session.
    // 5. Populate the pose of sessionâ€™s viewer reference space in referenceSpace at the time represented by frame into pose.
    auto populatePoseResult = populatePose(document, m_session->viewerReferenceSpace(), referenceSpace);
    if (populatePoseResult.hasException())
        return populatePoseResult.releaseException();

    // 6. If pose is null return null.
    auto populateValue = populatePoseResult.releaseReturnValue();
    if (!populateValue)
        return nullptr;

    RefPtr<WebXRViewerPose> pose = WebXRViewerPose::create(WebXRRigidTransform::create(populateValue->transform), populateValue->emulatedPosition);

    // 7. Let xrviews be an empty list.
    Vector<Ref<WebXRView>> xrViews;
    // 8. For each active view view in the list of views on session, perform the following steps:
    const auto& frameData = m_session->frameData();
    for (auto& view : m_session->views()) {
        auto index = xrViews.size();
        if (!view.active || frameData.views.size() <= index)
            continue;

        // 8.1 Let xrview be a new XRView object in the relevant realm of session.
        // 8.2 Initialize xrviewâ€™s underlying view to view.
        // 8.3 Initialize xrviewâ€™s eye to viewâ€™s eye.
        // 8.4 Initialize xrviewâ€™s frame time to frameâ€™s time.
        // 8.5 Initialize xrviewâ€™s session to session.
        // 8.6. Let offset be an new XRRigidTransform object equal to the view offset of view in the relevant realm of session.
        // 8.7. Set xrviewâ€™s transform property to the result of multiplying the XRViewerPose's transform by the offset transform in the relevant realm of session
        auto offset = matrixFromPose(frameData.views[index].offset);
        auto transform = WebXRRigidTransform::create(pose->transform().rawTransform() * offset);

        // Set projection matrix for each view
        std::array<float, 16> projection = WTF::switchOn(frameData.views[index].projection, [&](const PlatformXR::FrameData::Fov& fov) {
            double near = m_session->renderState().depthNear();
            double far = m_session->renderState().depthFar();
            return TransformationMatrix::fromProjection(fov.up, fov.down, fov.left, fov.right, near, far).toColumnMajorFloatArray();
        }, [&](const std::array<float, 16>& matrix) {
            return matrix;
        }, [&](const std::nullptr_t&) {
            // Use aspect projection for inline sessions
            double fov =  m_session->renderState().inlineVerticalFieldOfView().value_or(piOverTwoDouble);
            float aspect = 1;
            auto layer = m_session->renderState().baseLayer();
            if (layer)
                aspect = static_cast<double>(layer->framebufferWidth()) / static_cast<double>(layer->framebufferHeight());
            double near = m_session->renderState().depthNear();
            double far = m_session->renderState().depthFar();
            return TransformationMatrix::fromProjection(fov, aspect, near, far).toColumnMajorFloatArray();
        });

        auto xrView = WebXRView::create(Ref { *this }, view.eye, WTFMove(transform), Float32Array::create(projection.data(), projection.size()));
        xrView->setViewportModifiable(m_session->supportsViewportScaling());

        //  8.8. Append xrview to xrviews
        xrViews.append(WTFMove(xrView));
    }

    // 9. Set poseâ€™s views to xrviews
    pose->setViews(WTFMove(xrViews));

    // 10. Return pose.
    return pose;
}

ExceptionOr<RefPtr<WebXRPose>> WebXRFrame::getPose(const Document& document, const WebXRSpace& space, const WebXRSpace& baseSpace)
{
    // 1. Let frame be this.
    // 2. Let pose be a new XRPose object in the relevant realm of frame.
    // 3. Populate the pose of space in baseSpace at the time represented by frame into pose.

    auto populatePoseResult = populatePose(document, space, baseSpace);
    if (populatePoseResult.hasException())
        return populatePoseResult.releaseException();

    auto populateValue = populatePoseResult.releaseReturnValue();
    if (!populateValue)
        return nullptr;

    // 4. Return pose.
    return RefPtr<WebXRPose>(WebXRPose::create(WebXRRigidTransform::create(populateValue->transform), populateValue->emulatedPosition));
}

TransformationMatrix WebXRFrame::matrixFromPose(const PlatformXR::FrameData::Pose& pose)
{
    TransformationMatrix matrix;
    matrix.translate3d(pose.position.x(), pose.position.y(), pose.position.z());
    matrix.multiply(TransformationMatrix::fromQuaternion({ pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w }));
    return matrix;
}

#if ENABLE(WEBXR_HANDS)

// https://immersive-web.github.io/webxr-hand-input/#dom-xrframe-getjointpose
ExceptionOr<RefPtr<WebXRJointPose>> WebXRFrame::getJointPose(const Document& document, const WebXRJointSpace& jointSpace, const WebXRSpace& baseSpace)
{
    auto populatePoseResult = populatePose(document, jointSpace, baseSpace);
    if (populatePoseResult.hasException())
        return populatePoseResult.releaseException();

    auto populateValue = populatePoseResult.releaseReturnValue();
    if (!populateValue)
        return nullptr;

    return RefPtr<WebXRJointPose>(WebXRJointPose::create(WebXRRigidTransform::create(populateValue->transform), populateValue->emulatedPosition, jointSpace.radius()));
}

// https://immersive-web.github.io/webxr-hand-input/#dom-xrframe-filljointradii
ExceptionOr<bool> WebXRFrame::fillJointRadii(const Vector<Ref<WebXRJointSpace>>& jointSpaces, Float32Array& radii)
{
    // If frameâ€™s active boolean is false, throw an InvalidStateError and abort these steps.
    if (!m_active)
        return Exception { ExceptionCode::InvalidStateError, "Frame is not active"_s };

    // For each joint in the jointSpaces:
    // If jointâ€™s session is different from session, throw an InvalidStateError and abort these steps.
    for (const auto& jointSpace : jointSpaces) {
        if (jointSpace->session() != m_session.ptr())
            return Exception { ExceptionCode::InvalidStateError, "Joint space's session does not match frame's session"_s };
    }

    // If the length of jointSpaces is larger than the number of elements in radii, throw a TypeError and abort these steps.
    if (jointSpaces.size() > radii.length())
        return Exception { ExceptionCode::TypeError, "Unexpected length of radii array"_s };

    // Let allValid be true.
    bool allValid = true;

    // For each joint in the jointSpaces:
    // 1. Set the float value of radii at offset as follows:
    // If the user agent can determine the poses of all the joints belonging to the jointâ€™s hand:
    //     Set the float value of radii at offset to that radius.
    // Otherwise
    //     Set the float value of radii at offset to NaN.
    //     Set allValid to false.
    // 2. Increase offset by 1.
    for (size_t i = 0; i < jointSpaces.size(); ++i) {
        if (jointSpaces[i]->handHasMissingPoses()) {
            radii.set(i, std::numeric_limits<float>::quiet_NaN());
            allValid = false;
        } else
            radii.set(i, jointSpaces[i]->radius());
    }

    return allValid;
}

// https://immersive-web.github.io/webxr-hand-input/#dom-xrframe-fillposes
ExceptionOr<bool> WebXRFrame::fillPoses(const Document& document, const Vector<Ref<WebXRSpace>>& spaces, const WebXRSpace& baseSpace, Float32Array& transforms)
{
    // If frameâ€™s active boolean is false, throw an InvalidStateError and abort these steps.
    if (!m_active)
        return Exception { ExceptionCode::InvalidStateError, "Frame is not active"_s };

    // For each space in the spaces sequence:
    // If spaceâ€™s session is different from session, throw an InvalidStateError and abort these steps.
    for (const auto& space : spaces) {
        if (space->session() != m_session.ptr())
            return Exception { ExceptionCode::InvalidStateError, "Space's session does not match frame's session"_s };
    }

    // If baseSpaceâ€™s session is different from session, throw an InvalidStateError and abort these steps.
    if (baseSpace.session() != m_session.ptr())
        return Exception { ExceptionCode::InvalidStateError, "Base space's session does not match frame's session"_s };

    // If the length of spaces multiplied by 16 is larger than the number of elements in transforms,
    // throw a TypeError and abort these steps.
    const size_t numberOfFloatsPerTransform = 16;
    if (spaces.size() * numberOfFloatsPerTransform > transforms.length())
        return Exception { ExceptionCode::TypeError, "Unexpected length of transforms array"_s };

    // Check if poses may be reported and, if not, throw a SecurityError and abort these steps.
    if (!m_session->posesCanBeReported(document))
        return Exception { ExceptionCode::SecurityError, "Poses cannot be reported"_s };

    // Let allValid be true.
    bool allValid = true;

    // For each space in the spaces sequence:
    for (size_t spaceIndex = 0; spaceIndex < spaces.size(); ++spaceIndex) {
        // 1. Populate the pose of space in baseSpace at the time represented by frame into pose.
        auto populatePoseResult = populatePose(document, spaces[spaceIndex], baseSpace);
        if (populatePoseResult.hasException())
            return populatePoseResult.releaseException();

        // 2. If pose is null, perform the following steps:
        // 3. Set 16 consecutive elements of the transforms array starting at offset to NaN.
        // 4. Set allValid to false.
        auto populateValue = populatePoseResult.releaseReturnValue();
        if (!populateValue) {
            for (size_t transformIndex = 0; transformIndex < numberOfFloatsPerTransform; ++transformIndex)
                transforms.set(spaceIndex * numberOfFloatsPerTransform + transformIndex, std::numeric_limits<float>::quiet_NaN());
            allValid = false;
        } else {
            // 5. If pose is not null, copy all elements from poseâ€™s matrix member to the transforms array starting at offset.
            // 6. Increase offset by 16.
            auto matrix = populateValue->transform.toColumnMajorFloatArray();
            for (size_t transformIndex = 0; transformIndex < numberOfFloatsPerTransform; ++transformIndex)
                transforms.set(spaceIndex * numberOfFloatsPerTransform + transformIndex, matrix[transformIndex]);
        }
    }

    return allValid;
}

#endif

} // namespace WebCore

#endif // ENABLE(WEBXR)
