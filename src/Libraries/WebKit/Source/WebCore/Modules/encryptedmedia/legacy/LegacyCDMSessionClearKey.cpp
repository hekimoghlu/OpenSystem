/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include "LegacyCDMSessionClearKey.h"

#include "Logging.h"
#include "WebKitMediaKeyError.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/TypedArrayAdaptors.h>
#include <pal/text/TextEncoding.h>
#include <wtf/JSONValues.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UUID.h>
#include <wtf/text/Base64.h>

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)


namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CDMSessionClearKey);

CDMSessionClearKey::CDMSessionClearKey(LegacyCDMSessionClient& client)
    : m_client(client)
    , m_sessionId(createVersion4UUIDString())
{
}

CDMSessionClearKey::~CDMSessionClearKey() = default;

RefPtr<Uint8Array> CDMSessionClearKey::generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode)
{
    UNUSED_PARAM(mimeType);
    UNUSED_PARAM(destinationURL);
    UNUSED_PARAM(systemCode);

    if (!initData) {
        errorCode = WebKitMediaKeyError::MEDIA_KEYERR_CLIENT;
        return nullptr;
    }
    m_initData = initData;

    bool sawError = false;
    String keyID = PAL::UTF8Encoding().decode(m_initData->span(), true, sawError);
    if (sawError) {
        errorCode = WebKitMediaKeyError::MEDIA_KEYERR_CLIENT;
        return nullptr;
    }

    return initData;
}

void CDMSessionClearKey::releaseKeys()
{
    m_cachedKeys.clear();
}

bool CDMSessionClearKey::update(JSC::Uint8Array* rawKeysData, RefPtr<JSC::Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode)
{
    UNUSED_PARAM(nextMessage);
    UNUSED_PARAM(systemCode);
    ASSERT(rawKeysData);

    do {
        auto rawKeysString = String::fromUTF8(rawKeysData->span());
        if (rawKeysString.isEmpty())  {
            LOG(Media, "CDMSessionClearKey::update(%p) - failed: empty message", this);
            break;
        }

        auto keysDataValue = JSON::Value::parseJSON(rawKeysString);
        if (!keysDataValue || !keysDataValue->asObject()) {
            LOG(Media, "CDMSessionClearKey::update(%p) - failed: invalid JSON", this);
            break;
        }

        auto keysDataObject = keysDataValue->asObject();
        auto keysArray = keysDataObject->getArray("keys"_s);
        if (!keysArray) {
            LOG(Media, "CDMSessionClearKey::update(%p) - failed: keys array missing or empty", this);
            break;
        }

        auto length = keysArray->length();
        if (!length) {
            LOG(Media, "CDMSessionClearKey::update(%p) - failed: keys array missing or empty", this);
            break;
        }

        bool foundValidKey = false;
        for (unsigned i = 0; i < length; ++i) {
            auto keyObject = keysArray->get(i)->asObject();

            if (!keyObject) {
                LOG(Media, "CDMSessionClearKey::update(%p) - failed: null keyDictionary", this);
                continue;
            }

            auto getStringProperty = [&keyObject](ASCIILiteral name) -> String {
                String string;
                if (!keyObject->getString(name, string))
                    return { };
                return string;
            };

            auto algorithm = getStringProperty("alg"_s);
            if (!equalLettersIgnoringASCIICase(algorithm, "a128kw"_s)) {
                LOG(Media, "CDMSessionClearKey::update(%p) - failed: algorithm unsupported", this);
                continue;
            }

            auto keyType = getStringProperty("kty"_s);
            if (!equalLettersIgnoringASCIICase(keyType, "oct"_s)) {
                LOG(Media, "CDMSessionClearKey::update(%p) - failed: keyType unsupported", this);
                continue;
            }

            auto keyId = getStringProperty("kid"_s);
            if (keyId.isEmpty()) {
                LOG(Media, "CDMSessionClearKey::update(%p) - failed: keyId missing or empty", this);
                continue;
            }

            auto rawKeyData = getStringProperty("k"_s);
            if (rawKeyData.isEmpty())  {
                LOG(Media, "CDMSessionClearKey::update(%p) - failed: key missing or empty", this);
                continue;
            }

            auto keyData = base64Decode(rawKeyData);
            if (!keyData || keyData->isEmpty()) {
                LOG(Media, "CDMSessionClearKey::update(%p) - failed: unable to base64 decode key", this);
                continue;
            }

            m_cachedKeys.set(keyId, WTFMove(*keyData));
            foundValidKey = true;
        }

        if (foundValidKey)
            return true;

    } while (false);

    errorCode = WebKitMediaKeyError::MEDIA_KEYERR_CLIENT;
    return false;
}

RefPtr<JSC::ArrayBuffer> CDMSessionClearKey::cachedKeyForKeyID(const String& keyId) const
{
    if (!m_cachedKeys.contains(keyId))
        return nullptr;

    auto keyData = m_cachedKeys.get(keyId);
    auto keyDataArray = JSC::Uint8Array::create(keyData.data(), keyData.size());
    return keyDataArray->unsharedBuffer();
}

}

#endif
