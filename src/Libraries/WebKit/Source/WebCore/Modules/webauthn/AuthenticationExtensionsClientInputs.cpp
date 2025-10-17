/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
#include "AuthenticationExtensionsClientInputs.h"

#if ENABLE(WEB_AUTHN)

#include "BufferSource.h"
#include "CBORReader.h"
#include "CBORWriter.h"

namespace WebCore {

std::optional<AuthenticationExtensionsClientInputs> AuthenticationExtensionsClientInputs::fromCBOR(std::span<const uint8_t> buffer)
{
    std::optional<cbor::CBORValue> decodedValue = cbor::CBORReader::read(buffer);
    if (!decodedValue || !decodedValue->isMap())
        return std::nullopt;
    AuthenticationExtensionsClientInputs clientInputs;

    const auto& decodedMap = decodedValue->getMap();
    auto it = decodedMap.find(cbor::CBORValue("appid"));
    if (it != decodedMap.end() && it->second.isString())
        clientInputs.appid = it->second.getString();
    it = decodedMap.find(cbor::CBORValue("credProps"));
    if (it != decodedMap.end() && it->second.isBool())
        clientInputs.credProps = it->second.getBool();
    it = decodedMap.find(cbor::CBORValue("largeBlob"));
    if (it != decodedMap.end() && it->second.isMap()) {
        const auto& largeBlobMap = it->second.getMap();
        AuthenticationExtensionsClientInputs::LargeBlobInputs largeBlob;
        auto largeBlobIt = largeBlobMap.find(cbor::CBORValue("support"));

        if (largeBlobIt != largeBlobMap.end() && largeBlobIt->second.isString())
            largeBlob.support = largeBlobIt->second.getString();

        largeBlobIt = largeBlobMap.find(cbor::CBORValue("read"));
        if (largeBlobIt != largeBlobMap.end() && largeBlobIt->second.isBool())
            largeBlob.read = largeBlobIt->second.getBool();

        largeBlobIt = largeBlobMap.find(cbor::CBORValue("write"));
        if (largeBlobIt != largeBlobMap.end() && largeBlobIt->second.isByteString()) {
            RefPtr<ArrayBuffer> write = ArrayBuffer::create(largeBlobIt->second.getByteString());
            largeBlob.write = BufferSource(write);
        }

        clientInputs.largeBlob = largeBlob;
    }

    return clientInputs;
}

Vector<uint8_t> AuthenticationExtensionsClientInputs::toCBOR() const
{
    cbor::CBORValue::MapValue clientInputsMap;
    if (!appid.isEmpty())
        clientInputsMap[cbor::CBORValue("appid")] = cbor::CBORValue(appid);
    if (credProps)
        clientInputsMap[cbor::CBORValue("credProps")] = cbor::CBORValue(*credProps);
    if (largeBlob) {
        cbor::CBORValue::MapValue largeBlobMap;
        if (!largeBlob->support.isNull())
            largeBlobMap[cbor::CBORValue("support")] = cbor::CBORValue(largeBlob->support);

        if (largeBlob->read)
            largeBlobMap[cbor::CBORValue("read")] = cbor::CBORValue(largeBlob->read.value());

        if (largeBlob->write)
            largeBlobMap[cbor::CBORValue("write")] = cbor::CBORValue(largeBlob->write.value());

        clientInputsMap[cbor::CBORValue("largeBlob")] = cbor::CBORValue(largeBlobMap);
    }

    auto clientInputs = cbor::CBORWriter::write(cbor::CBORValue(WTFMove(clientInputsMap)));
    ASSERT(clientInputs);

    return *clientInputs;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
