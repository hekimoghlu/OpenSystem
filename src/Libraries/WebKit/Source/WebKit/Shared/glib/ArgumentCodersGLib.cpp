/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#include "ArgumentCodersGLib.h"

#include <gio/gio.h>
#include <gio/gunixfdlist.h>
#include <wtf/Vector.h>
#include <wtf/glib/GSpanExtras.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

namespace IPC {

void ArgumentCoder<GRefPtr<GByteArray>>::encode(Encoder& encoder, const GRefPtr<GByteArray>& array)
{
    if (!array) {
        encoder << false;
        return;
    }

    encoder << true;
    encoder << span(array);
}

std::optional<GRefPtr<GByteArray>> ArgumentCoder<GRefPtr<GByteArray>>::decode(Decoder& decoder)
{
    auto isEngaged = decoder.decode<bool>();
    if (!UNLIKELY(isEngaged))
        return std::nullopt;

    if (!(*isEngaged))
        return GRefPtr<GByteArray>();

    auto data = decoder.decode<std::span<const uint8_t>>();
    if (UNLIKELY(!data))
        return std::nullopt;

    GRefPtr<GByteArray> array = adoptGRef(g_byte_array_sized_new(data->size()));
    g_byte_array_append(array.get(), data->data(), data->size());

    return array;
}

void ArgumentCoder<GRefPtr<GVariant>>::encode(Encoder& encoder, const GRefPtr<GVariant>& variant)
{
    if (!variant) {
        encoder << CString();
        return;
    }

    encoder << CString(g_variant_get_type_string(variant.get()));
    encoder << span(variant);
}

std::optional<GRefPtr<GVariant>> ArgumentCoder<GRefPtr<GVariant>>::decode(Decoder& decoder)
{
    auto variantTypeString = decoder.decode<CString>();
    if (UNLIKELY(!variantTypeString))
        return std::nullopt;

    if (variantTypeString->isNull())
        return GRefPtr<GVariant>();

    if (!g_variant_type_string_is_valid(variantTypeString->data()))
        return std::nullopt;

    auto data = decoder.decode<std::span<const uint8_t>>();
    if (UNLIKELY(!data))
        return std::nullopt;

    GUniquePtr<GVariantType> variantType(g_variant_type_new(variantTypeString->data()));
    GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new(data->data(), data->size()));
    return g_variant_new_from_bytes(variantType.get(), bytes.get(), FALSE);
}

void ArgumentCoder<GRefPtr<GTlsCertificate>>::encode(Encoder& encoder, const GRefPtr<GTlsCertificate>& certificate)
{
    if (!certificate) {
        encoder << Vector<std::span<const uint8_t>> { };
        return;
    }

    Vector<GRefPtr<GByteArray>> certificatesData;
    for (auto* nextCertificate = certificate.get(); nextCertificate; nextCertificate = g_tls_certificate_get_issuer(nextCertificate)) {
        GRefPtr<GByteArray> certificateData;
        g_object_get(nextCertificate, "certificate", &certificateData.outPtr(), nullptr);

        if (!certificateData) {
            certificatesData.clear();
            break;
        }
        certificatesData.insert(0, WTFMove(certificateData));
    }

    if (certificatesData.isEmpty())
        return;

    encoder << certificatesData;

    GRefPtr<GByteArray> privateKey;
    GUniqueOutPtr<char> privateKeyPKCS11Uri;
    g_object_get(certificate.get(), "private-key", &privateKey.outPtr(), "private-key-pkcs11-uri", &privateKeyPKCS11Uri.outPtr(), nullptr);
    encoder << privateKey;
    encoder << CString(privateKeyPKCS11Uri.get());
}

std::optional<GRefPtr<GTlsCertificate>> ArgumentCoder<GRefPtr<GTlsCertificate>>::decode(Decoder& decoder)
{
    auto certificatesData = decoder.decode<WTF::Vector<GRefPtr<GByteArray>>>();

    if (UNLIKELY(!certificatesData))
        return std::nullopt;

    if (!certificatesData->size())
        return GRefPtr<GTlsCertificate>();

    std::optional<GRefPtr<GByteArray>> privateKey;
    decoder >> privateKey;
    if (UNLIKELY(!privateKey))
        return std::nullopt;

    std::optional<CString> privateKeyPKCS11Uri;
    decoder >> privateKeyPKCS11Uri;
    if (UNLIKELY(!privateKeyPKCS11Uri))
        return std::nullopt;

    GType certificateType = g_tls_backend_get_certificate_type(g_tls_backend_get_default());
    GRefPtr<GTlsCertificate> certificate;
    GTlsCertificate* issuer = nullptr;
    for (uint32_t i = 0; auto& certificateData : *certificatesData) {
        certificate = adoptGRef(G_TLS_CERTIFICATE(g_initable_new(
            certificateType, nullptr, nullptr,
            "certificate", certificateData.get(),
            "issuer", issuer,
            "private-key", i == certificatesData->size() - 1 ? privateKey->get() : nullptr,
            "private-key-pkcs11-uri", i == certificatesData->size() - 1 ? privateKeyPKCS11Uri->data() : nullptr,
            nullptr)));
        issuer = certificate.get();
    }

    return certificate;
}

void ArgumentCoder<GTlsCertificateFlags>::encode(Encoder& encoder, GTlsCertificateFlags flags)
{
    encoder << static_cast<uint32_t>(flags);
}

std::optional<GTlsCertificateFlags> ArgumentCoder<GTlsCertificateFlags>::decode(Decoder& decoder)
{
    auto flags = decoder.decode<uint32_t>();
    if (UNLIKELY(!flags))
        return std::nullopt;
    return static_cast<GTlsCertificateFlags>(*flags);
}

void ArgumentCoder<GRefPtr<GUnixFDList>>::encode(Encoder& encoder, const GRefPtr<GUnixFDList>& fdList)
{
    if (!fdList) {
        encoder << false;
        return;
    }

    Vector<UnixFileDescriptor> attachments;
    unsigned length = std::max(0, g_unix_fd_list_get_length(fdList.get()));
    if (length) {
        attachments = Vector<UnixFileDescriptor>(length, [&](size_t i) {
            return UnixFileDescriptor { g_unix_fd_list_get(fdList.get(), i, nullptr), UnixFileDescriptor::Adopt };
        });
    }
    encoder << true << WTFMove(attachments);
}

std::optional<GRefPtr<GUnixFDList>> ArgumentCoder<GRefPtr<GUnixFDList>>::decode(Decoder& decoder)
{
    auto hasObject = decoder.decode<bool>();
    if (UNLIKELY(!hasObject))
        return std::nullopt;
    if (!*hasObject)
        return GRefPtr<GUnixFDList> { };

    auto attachments = decoder.decode<Vector<UnixFileDescriptor>>();
    if (UNLIKELY(!attachments))
        return std::nullopt;

    GRefPtr<GUnixFDList> fdList = adoptGRef(g_unix_fd_list_new());
    for (auto& attachment : *attachments) {
        int ret = g_unix_fd_list_append(fdList.get(), attachment.value(), nullptr);
        if (UNLIKELY(ret == -1))
            return std::nullopt;
    }
    return fdList;
}

} // namespace IPC
