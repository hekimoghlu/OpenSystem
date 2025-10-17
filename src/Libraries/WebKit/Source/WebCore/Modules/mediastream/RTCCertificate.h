/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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

#if ENABLE(WEB_RTC)

#include "SecurityOrigin.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCCertificate : public RefCounted<RTCCertificate> {
public:
    struct DtlsFingerprint {
        String algorithm;
        String value;
    };

    static Ref<RTCCertificate> create(Ref<SecurityOrigin>&&, double expires, Vector<DtlsFingerprint>&&, String&& pemCertificate, String&& pemPrivateKey);

    double expires() const { return m_expires; }
    const Vector<DtlsFingerprint>& getFingerprints() const { return m_fingerprints; }

    const String& pemCertificate() const { return m_pemCertificate; }
    const String& pemPrivateKey() const { return m_pemPrivateKey; }
    const SecurityOrigin& origin() const { return m_origin.get(); }

private:
    RTCCertificate(Ref<SecurityOrigin>&&, double expires, Vector<DtlsFingerprint>&&, String&& pemCertificate, String&& pemPrivateKey);

    Ref<SecurityOrigin> m_origin;
    double m_expires { 0 };
    Vector<DtlsFingerprint> m_fingerprints;
    String m_pemCertificate;
    String m_pemPrivateKey;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
