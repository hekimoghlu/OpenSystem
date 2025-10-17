/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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

import Foundation

// Apple TVs and watches have no UI to enable or disable this status.
// So, help them out by ignoring all efforts.
extension TPPBPeerStableInfoUserControllableViewStatus {
    func sanitizeForPlatform(permanentInfo: TPPeerPermanentInfo) -> TPPBPeerStableInfoUserControllableViewStatus {
        // Unknown is the unknown for any platform
        if self == .UNKNOWN {
            return .UNKNOWN
        }

        if permanentInfo.modelID.hasPrefix("AppleTV") ||
            permanentInfo.modelID.hasPrefix("AudioAccessory") {
            // Apple TVs, and HomePods don't have UI to set this bit. So, they should always sync the
            // user-controlled views to which they have access.
            //
            // Some watches don't have UI to set the bit, but some do.
            //
            // Note that we want this sanitization behavior to be baked into the local OS, which is what owns
            // the UI software, and not in the Policy, which can change.
            return .FOLLOWING
        } else {
            // All other platforms can choose their own fate
            return self
        }
    }
}

extension StableChanges {
    static func change(viewStatus: TPPBPeerStableInfoUserControllableViewStatus?) -> StableChanges? {
        if viewStatus == nil {
            return nil
        }
        return StableChanges(deviceName: nil,
                             serialNumber: nil,
                             osVersion: nil,
                             policyVersion: nil,
                             policySecrets: nil,
                             setSyncUserControllableViews: viewStatus,
                             secureElementIdentity: nil,
                             walrusSetting: nil,
                             webAccess: nil)
    }
}
