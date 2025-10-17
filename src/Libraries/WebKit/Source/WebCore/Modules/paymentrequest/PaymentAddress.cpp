/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#include "PaymentAddress.h"

#if ENABLE(PAYMENT_REQUEST)

namespace WebCore {

PaymentAddress::PaymentAddress(const String& country, const Vector<String>& addressLine, const String& region, const String& city, const String& dependentLocality, const String& postalCode, const String& sortingCode, const String& organization, const String& recipient, const String& phone)
    : m_country { country }
    , m_addressLine { addressLine }
    , m_region { region }
    , m_city { city }
    , m_dependentLocality { dependentLocality }
    , m_postalCode { postalCode }
    , m_sortingCode { sortingCode }
    , m_organization { organization }
    , m_recipient { recipient }
    , m_phone { phone }
{
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
