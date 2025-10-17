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
#pragma once

#if ENABLE(PAYMENT_REQUEST)

#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class PaymentAddress final : public RefCounted<PaymentAddress> {
public:
    template <typename... Args> static Ref<PaymentAddress> create(Args&&... args)
    {
        return adoptRef(*new PaymentAddress(std::forward<Args>(args)...));
    }

    const String& country() const { return m_country; }
    const Vector<String>& addressLine() const { return m_addressLine; }
    const String& region() const { return m_region; }
    const String& city() const { return m_city; }
    const String& dependentLocality() const { return m_dependentLocality; }
    const String& postalCode() const { return m_postalCode; }
    const String& sortingCode() const { return m_sortingCode; }
    const String& organization() const { return m_organization; }
    const String& recipient() const { return m_recipient; }
    const String& phone() const { return m_phone; }

private:
    PaymentAddress(const String& country, const Vector<String>& addressLine, const String& region, const String& city, const String& dependentLocality, const String& postalCode, const String& sortingCode, const String& organization, const String& recipient, const String& phone);

    String m_country;
    Vector<String> m_addressLine;
    String m_region;
    String m_city;
    String m_dependentLocality;
    String m_postalCode;
    String m_sortingCode;
    String m_organization;
    String m_recipient;
    String m_phone;
};

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
