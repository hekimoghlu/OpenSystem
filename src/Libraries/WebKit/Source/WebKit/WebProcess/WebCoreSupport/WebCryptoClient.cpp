/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#include "WebCryptoClient.h"

#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include "WebProcessProxyMessages.h"
#include <WebCore/SerializedCryptoKeyWrap.h>
#include <WebCore/WrappedCryptoKey.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebCryptoClient);

std::optional<Vector<uint8_t>> WebCryptoClient::wrapCryptoKey(const Vector<uint8_t>& key) const
{
    if (m_pageIdentifier) {
        auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebPageProxy::WrapCryptoKey(key), *m_pageIdentifier);
        auto [wrappedKey] = sendResult.takeReplyOr(std::nullopt);
        return wrappedKey;
    }
    auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebProcessProxy::WrapCryptoKey(key), 0);

    auto [wrappedKey] = sendResult.takeReplyOr(std::nullopt);
    return wrappedKey;
}

std::optional<Vector<uint8_t>> WebCryptoClient::serializeAndWrapCryptoKey(WebCore::CryptoKeyData&& keyData) const
{
    if (m_pageIdentifier) {
        auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebPageProxy::SerializeAndWrapCryptoKey(WTFMove(keyData)), *m_pageIdentifier);
        auto [wrappedKey] = sendResult.takeReplyOr(std::nullopt);
        return wrappedKey;
    }
    auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebProcessProxy::SerializeAndWrapCryptoKey(WTFMove(keyData)), 0);

    auto [wrappedKey] = sendResult.takeReplyOr(std::nullopt);
    return wrappedKey;
}

std::optional<Vector<uint8_t>> WebCryptoClient::unwrapCryptoKey(const Vector<uint8_t>& wrappedKey) const
{
    auto deserializedKey = WebCore::readSerializedCryptoKey(wrappedKey);
    if (!deserializedKey)
        return std::nullopt;

    if (m_pageIdentifier) {
        auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebPageProxy::UnwrapCryptoKey(*deserializedKey), *m_pageIdentifier);
        auto [unwrappedKey] = sendResult.takeReplyOr(std::nullopt);
        return unwrappedKey;
    }

    auto sendResult = WebProcess::singleton().parentProcessConnection()->sendSync(Messages::WebProcessProxy::UnwrapCryptoKey(*deserializedKey), 0);
    auto [unwrappedKey] = sendResult.takeReplyOr(std::nullopt);
    return unwrappedKey;
}

WebCryptoClient::WebCryptoClient(WebCore::PageIdentifier pageIdentifier)
    : m_pageIdentifier(pageIdentifier)
{
}
}
