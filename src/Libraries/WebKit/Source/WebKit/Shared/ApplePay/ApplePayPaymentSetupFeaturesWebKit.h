/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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

OBJC_CLASS NSArray;
OBJC_CLASS PKPaymentSetupFeature;

#include <WebCore/ApplePaySetupFeatureWebCore.h>
#include <wtf/ArgumentCoder.h>
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class PaymentSetupFeatures {
public:
    PaymentSetupFeatures(Vector<Ref<WebCore::ApplePaySetupFeature>>&&);
    PaymentSetupFeatures(RetainPtr<NSArray>&& = nullptr);

    NSArray *platformFeatures() const { return m_platformFeatures.get(); }
    operator Vector<Ref<WebCore::ApplePaySetupFeature>>() const;

private:
    friend struct IPC::ArgumentCoder<PaymentSetupFeatures, void>;
    RetainPtr<NSArray> m_platformFeatures;
};

} // namespace WebKit

#endif // ENABLE(APPLE_PAY)
