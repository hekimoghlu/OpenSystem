/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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

#include "Allowlist.h"
#include <wtf/HashSet.h>
#include <wtf/HashTraits.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class Allowlist;
class Document;
class HTMLFrameOwnerElement;
class HTMLIFrameElement;
struct OwnerPermissionsPolicyData;

class PermissionsPolicy {
    WTF_MAKE_TZONE_ALLOCATED(PermissionsPolicy);
public:
    PermissionsPolicy();
    PermissionsPolicy(const Document&);

    enum class Feature : uint8_t {
        Camera = 0,
        Microphone,
        SpeakerSelection,
        DisplayCapture,
        Gamepad,
        Geolocation,
        Payment,
        ScreenWakeLock,
        SyncXHR,
        Fullscreen,
        WebShare,
#if ENABLE(DEVICE_ORIENTATION)
        Gyroscope,
        Accelerometer,
        Magnetometer,
#endif
#if ENABLE(WEB_AUTHN)
        PublickeyCredentialsGetRule,
        DigitalCredentialsGetRule,
#endif
#if ENABLE(WEBXR)
        XRSpatialTracking,
#endif
        PrivateToken,
        Invalid
    };
    enum class ShouldReportViolation : bool { No, Yes };
    static bool isFeatureEnabled(Feature, const Document&, ShouldReportViolation = ShouldReportViolation::Yes);
    bool inheritedPolicyValueForFeature(Feature) const;

    // InheritedPolicy contains enabled features.
    using InheritedPolicy = HashSet<Feature, IntHash<Feature>, WTF::StrongEnumHashTraits<Feature>>;
    PermissionsPolicy(const InheritedPolicy& inheritedPolicy)
        : m_inheritedPolicy(inheritedPolicy)
    {
    }
    InheritedPolicy inheritedPolicy() const { return m_inheritedPolicy; }

    // https://w3c.github.io/webappsec-permissions-policy/#policy-directives
    using PolicyDirective = HashMap<Feature, Allowlist, IntHash<Feature>, WTF::StrongEnumHashTraits<Feature>>;
    static PolicyDirective processPermissionsPolicyAttribute(const HTMLIFrameElement&);

private:
    bool computeInheritedPolicyValueInContainer(Feature, const std::optional<OwnerPermissionsPolicyData>&, const SecurityOriginData&) const;

    InheritedPolicy m_inheritedPolicy;
};

} // namespace WebCore
