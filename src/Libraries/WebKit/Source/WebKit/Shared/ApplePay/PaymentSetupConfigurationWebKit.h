/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

#include <WebCore/ApplePaySetupConfiguration.h>
#include <wtf/RetainPtr.h>
#include <wtf/URL.h>

OBJC_CLASS PKPaymentSetupConfiguration;

namespace WebKit {

class PaymentSetupConfiguration {
public:
    PaymentSetupConfiguration(const WebCore::ApplePaySetupConfiguration&, const URL&);

    RetainPtr<PKPaymentSetupConfiguration> platformConfiguration() const;

    const WebCore::ApplePaySetupConfiguration& configuration() const { return m_configuration; }
    const URL& url() const { return m_url; }

private:
    WebCore::ApplePaySetupConfiguration m_configuration;
    URL m_url;
};

} // namespace WebKit

#endif // ENABLE(APPLE_PAY)

