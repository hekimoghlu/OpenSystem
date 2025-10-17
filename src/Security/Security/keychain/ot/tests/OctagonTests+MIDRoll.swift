/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

import FeatureFlags
import Foundation

class OctagonTestsMidRoll: OctagonTestsBase {
    struct Key: FeatureFlagsKey {
        let domain: StaticString
        let feature: StaticString
    }

    static func isEnabled() -> Bool {
        return isFeatureEnabled(Key(domain: "Security", feature: "RollIdentityOnMIDRotation"))
    }

    func testFeatureFlag() throws {
        XCTAssertFalse(OctagonTestsMidRoll.isEnabled(), "feature flag should be disabled")
    }

    func testFeatureFlagOverride () throws {
        XCTAssertFalse(IsRollOctagonIdentityEnabled(), "feaature flag should be disabled")

        SetRollOctagonIdentityEnabled(true)
        XCTAssertTrue(IsRollOctagonIdentityEnabled(), "feaature flag should be enabled")

        ClearRollOctagonIdentityEnabledOverride()
        XCTAssertFalse(IsRollOctagonIdentityEnabled(), "feaature flag should be disabled")

        SetRollOctagonIdentityEnabled(true)
        XCTAssertTrue(IsRollOctagonIdentityEnabled(), "feaature flag should be enabled")

        SetRollOctagonIdentityEnabled(false)
        XCTAssertFalse(IsRollOctagonIdentityEnabled(), "feaature flag should be disabled")
    }
}
