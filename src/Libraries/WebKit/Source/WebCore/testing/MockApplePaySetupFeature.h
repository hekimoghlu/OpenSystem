/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

#if ENABLE(APPLE_PAY)

#include "ApplePaySetupFeatureWebCore.h"

namespace WebCore {

class MockApplePaySetupFeature final : public ApplePaySetupFeature {
public:
    static Ref<MockApplePaySetupFeature> create(ApplePaySetupFeatureState, ApplePaySetupFeatureType, bool supportsInstallments);
    
    ApplePaySetupFeatureState state() const final { return m_state; }
    ApplePaySetupFeatureType type() const final { return m_type; }

#if ENABLE(APPLE_PAY_INSTALLMENTS)
    bool supportsInstallments() const final { return m_supportsInstallments; }
#endif

private:
    MockApplePaySetupFeature(ApplePaySetupFeatureState, ApplePaySetupFeatureType, bool supportsInstallments);

    ApplePaySetupFeatureState m_state;
    ApplePaySetupFeatureType m_type;
#if ENABLE(APPLE_PAY_INSTALLMENTS)
    bool m_supportsInstallments;
#endif
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
