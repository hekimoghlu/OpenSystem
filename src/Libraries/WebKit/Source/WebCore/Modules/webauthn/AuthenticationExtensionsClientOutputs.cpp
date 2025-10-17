/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#include "AuthenticationExtensionsClientOutputs.h"

#if ENABLE(WEB_AUTHN)

#include "CBORReader.h"
#include "CBORWriter.h"
#include "CredentialPropertiesOutput.h"
#include <wtf/text/Base64.h>

namespace WebCore {

std::optional<AuthenticationExtensionsClientOutputs> AuthenticationExtensionsClientOutputs::fromCBOR(const Vector<uint8_t>& buffer)
{
    std::optional<cbor::CBORValue> decodedValue = cbor::CBORReader::read(buffer);
    if (!decodedValue || !decodedValue->isMap())
        return std::nullopt;
    AuthenticationExtensionsClientOutputs clientOutputs;

    const auto& decodedMap = decodedValue->getMap();
    auto it = decodedMap.find(cbor::CBORValue("appid"));
    if (it != decodedMap.end() && it->second.isBool())
        clientOutputs.appid = it->second.getBool();
    it = decodedMap.find(cbor::CBORValue("credProps"));
    if (it != decodedMap.end() && it->second.isMap()) {
        CredentialPropertiesOutput credProps;
        it = it->second.getMap().find(cbor::CBORValue("rk"));
        if (it != decodedMap.end() && it->second.isBool())
            credProps.rk = it->second.getBool();
        clientOutputs.credProps = credProps;
    }

    it = decodedMap.find(cbor::CBORValue("largeBlob"));
    if (it != decodedMap.end() && it->second.isMap()) {
        const auto& largeBlobMap = it->second.getMap();
        LargeBlobOutputs largeBlob;

        auto largeBlobIt = largeBlobMap.find(cbor::CBORValue("supported"));
        if (largeBlobIt != largeBlobMap.end() && largeBlobIt->second.isBool())
            largeBlob.supported = largeBlobIt->second.getBool();

        largeBlobIt = largeBlobMap.find(cbor::CBORValue("blob"));
        if (largeBlobIt != largeBlobMap.end() && largeBlobIt->second.isByteString()) {
            RefPtr<ArrayBuffer> blob = ArrayBuffer::create(largeBlobIt->second.getByteString());
            largeBlob.blob = WTFMove(blob);
        }

        largeBlobIt = largeBlobMap.find(cbor::CBORValue("written"));
        if (largeBlobIt != largeBlobMap.end() && largeBlobIt->second.isBool())
            largeBlob.written = largeBlobIt->second.getBool();

        clientOutputs.largeBlob = largeBlob;
    }

    return clientOutputs;
}

Vector<uint8_t> AuthenticationExtensionsClientOutputs::toCBOR() const
{
    cbor::CBORValue::MapValue clientOutputsMap;
    if (appid)
        clientOutputsMap[cbor::CBORValue("appid")] = cbor::CBORValue(*appid);
    if (credProps) {
        cbor::CBORValue::MapValue credPropsMap;
        credPropsMap[cbor::CBORValue("rk")] = cbor::CBORValue(credProps->rk);
        clientOutputsMap[cbor::CBORValue("credProps")] = cbor::CBORValue(credPropsMap);
    }

    if (largeBlob) {
        cbor::CBORValue::MapValue largeBlobMap;
        if (largeBlob->supported)
            largeBlobMap[cbor::CBORValue("supported")] = cbor::CBORValue(largeBlob->supported.value());

        if (largeBlob->blob)
            largeBlobMap[cbor::CBORValue("blob")] = cbor::CBORValue(largeBlob->blob->toVector());

        if (largeBlob->written)
            largeBlobMap[cbor::CBORValue("written")] = cbor::CBORValue(largeBlob->written.value());

        clientOutputsMap[cbor::CBORValue("largeBlob")] = cbor::CBORValue(largeBlobMap);
    }

    auto clientOutputs = cbor::CBORWriter::write(cbor::CBORValue(WTFMove(clientOutputsMap)));
    ASSERT(clientOutputs);

    return *clientOutputs;
}

AuthenticationExtensionsClientOutputsJSON AuthenticationExtensionsClientOutputs::toJSON() const
{
    AuthenticationExtensionsClientOutputsJSON result;
    result.appid = appid;
    result.credProps = credProps;
    if (largeBlob) {
        result.largeBlob = AuthenticationExtensionsClientOutputsJSON::LargeBlobOutputsJSON {
            largeBlob->supported,
            base64URLEncodeToString(largeBlob->blob->span()),
            largeBlob->written,
        };
    }
    if (prf) {
        result.prf = AuthenticationExtensionsClientOutputsJSON::PRFOutputsJSON {
            prf->enabled,
            AuthenticationExtensionsClientOutputsJSON::PRFValuesJSON {
                base64URLEncodeToString(largeBlob->blob->span()),
                base64URLEncodeToString(largeBlob->blob->span()),
            },
        };
    }
    return result;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
