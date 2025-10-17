/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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

#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS PKPaymentSetupFeature;

namespace WebCore {

enum class ApplePaySetupFeatureState : uint8_t;
enum class ApplePaySetupFeatureType : bool;

class ApplePaySetupFeature : public RefCounted<ApplePaySetupFeature> {
public:
    static Ref<ApplePaySetupFeature> create(PKPaymentSetupFeature *feature)
    {
        return adoptRef(*new ApplePaySetupFeature(feature));
    }

    WEBCORE_EXPORT static bool supportsFeature(PKPaymentSetupFeature *);
    
    WEBCORE_EXPORT virtual ~ApplePaySetupFeature();

    virtual ApplePaySetupFeatureState state() const;
    virtual ApplePaySetupFeatureType type() const;

    PKPaymentSetupFeature *platformFeature() const { return m_feature.get(); }

#if ENABLE(APPLE_PAY_INSTALLMENTS)
    virtual bool supportsInstallments() const;
#endif

protected:
    WEBCORE_EXPORT ApplePaySetupFeature();

private:
    WEBCORE_EXPORT explicit ApplePaySetupFeature(PKPaymentSetupFeature *);

    RetainPtr<PKPaymentSetupFeature> m_feature;
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
