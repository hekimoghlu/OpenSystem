/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#import "CSPExtensionUtilities.h"

namespace WebKit {

_WKContentSecurityPolicyModeForExtension toWKContentSecurityPolicyModeForExtension(WebCore::ContentSecurityPolicyModeForExtension mode)
{
    switch (mode) {
    case WebCore::ContentSecurityPolicyModeForExtension::None:
        return _WKContentSecurityPolicyModeForExtensionNone;
    case WebCore::ContentSecurityPolicyModeForExtension::ManifestV2:
        return _WKContentSecurityPolicyModeForExtensionManifestV2;
    case WebCore::ContentSecurityPolicyModeForExtension::ManifestV3:
        return _WKContentSecurityPolicyModeForExtensionManifestV3;
    }
    return _WKContentSecurityPolicyModeForExtensionNone;
}

WebCore::ContentSecurityPolicyModeForExtension toContentSecurityPolicyModeForExtension(_WKContentSecurityPolicyModeForExtension wkMode)
{
    WebCore::ContentSecurityPolicyModeForExtension mode;
    switch (wkMode) {
    case _WKContentSecurityPolicyModeForExtensionNone:
        mode = WebCore::ContentSecurityPolicyModeForExtension::None;
        break;
    case _WKContentSecurityPolicyModeForExtensionManifestV2:
        mode = WebCore::ContentSecurityPolicyModeForExtension::ManifestV2;
        break;
    case _WKContentSecurityPolicyModeForExtensionManifestV3:
        mode = WebCore::ContentSecurityPolicyModeForExtension::ManifestV3;
        break;
    }

    return mode;
}

} // namespace WebKit
