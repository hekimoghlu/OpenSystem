/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#include "LibWebRTCCertificateGenerator.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "LibWebRTCProvider.h"
#include "LibWebRTCUtils.h"
#include "RTCCertificate.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/rtc_base/ref_counted_object.h>
#include <webrtc/rtc_base/rtc_certificate_generator.h>
#include <webrtc/rtc_base/ssl_certificate.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

namespace LibWebRTCCertificateGenerator {

class RTCCertificateGeneratorCallbackWrapper : public ThreadSafeRefCounted<RTCCertificateGeneratorCallbackWrapper, WTF::DestructionThread::Main> {
public:
    static Ref<RTCCertificateGeneratorCallbackWrapper> create(Ref<SecurityOrigin>&& origin, Function<void(ExceptionOr<Ref<RTCCertificate>>&&)>&& resultCallback)
    {
        return adoptRef(*new RTCCertificateGeneratorCallbackWrapper(WTFMove(origin), WTFMove(resultCallback)));
    }

    void process(rtc::scoped_refptr<rtc::RTCCertificate> certificate)
    {
        callOnMainThread([origin = m_origin.releaseNonNull(), callback = WTFMove(m_resultCallback), certificate = WTFMove(certificate)]() mutable {
            if (!certificate) {
                callback(Exception { ExceptionCode::TypeError, "Unable to create a certificate"_s });
                return;
            }

            Vector<RTCCertificate::DtlsFingerprint> fingerprints;
            auto stats = certificate->GetSSLCertificate().GetStats();
            auto* info = stats.get();
            while (info) {
                StringView fingerprint { std::span { info->fingerprint } };
                fingerprints.append({ fromStdString(info->fingerprint_algorithm), fingerprint.convertToASCIILowercase() });
                info = info->issuer.get();
            };

            auto pem = certificate->ToPEM();
            callback(RTCCertificate::create(WTFMove(origin), certificate->Expires(), WTFMove(fingerprints), fromStdString(pem.certificate()), fromStdString(pem.private_key())));
        });
    }

private:
    RTCCertificateGeneratorCallbackWrapper(Ref<SecurityOrigin>&& origin, Function<void(ExceptionOr<Ref<RTCCertificate>>&&)>&& resultCallback)
        : m_origin(WTFMove(origin))
        , m_resultCallback(WTFMove(resultCallback))
    {
    }

    RefPtr<SecurityOrigin> m_origin;
    Function<void(ExceptionOr<Ref<RTCCertificate>>&&)> m_resultCallback;
};

static inline rtc::KeyParams keyParamsFromCertificateType(const PeerConnectionBackend::CertificateInformation& info)
{
    switch (info.type) {
    case PeerConnectionBackend::CertificateInformation::Type::ECDSAP256:
        return rtc::KeyParams::ECDSA();
    case PeerConnectionBackend::CertificateInformation::Type::RSASSAPKCS1v15:
        if (info.rsaParameters)
            return rtc::KeyParams::RSA(info.rsaParameters->modulusLength, info.rsaParameters->publicExponent);
        return rtc::KeyParams::RSA(2048, 65537);
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void generateCertificate(Ref<SecurityOrigin>&& origin, LibWebRTCProvider& provider, const PeerConnectionBackend::CertificateInformation& info, Function<void(ExceptionOr<Ref<RTCCertificate>>&&)>&& resultCallback)
{
    auto callbackWrapper = RTCCertificateGeneratorCallbackWrapper::create(WTFMove(origin), WTFMove(resultCallback));

    std::optional<uint64_t> expiresMs;
    if (info.expires)
        expiresMs = static_cast<uint64_t>(*info.expires);

    provider.prepareCertificateGenerator([info, expiresMs, callbackWrapper = WTFMove(callbackWrapper)](auto& generator) mutable {
        generator.GenerateCertificateAsync(keyParamsFromCertificateType(info), expiresMs, [callbackWrapper = WTFMove(callbackWrapper)](rtc::scoped_refptr<rtc::RTCCertificate> certificate) mutable {
            callbackWrapper->process(WTFMove(certificate));
        });
    });
}

} // namespace LibWebRTCCertificateGenerator

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
