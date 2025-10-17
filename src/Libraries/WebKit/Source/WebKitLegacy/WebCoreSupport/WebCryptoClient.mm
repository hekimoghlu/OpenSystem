/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#import "WebCryptoClient.h"

#import "WebDelegateImplementationCaching.h"
#import "WebFramePrivate.h"
#import "WebUIDelegatePrivate.h"
#import <WebCore/CryptoKey.h>
#import <WebCore/SerializedCryptoKeyWrap.h>
#import <WebCore/SerializedScriptValue.h>
#import <WebCore/WrappedCryptoKey.h>
#import <optional>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/VectorCocoa.h>

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebCryptoClient);

std::optional<Vector<uint8_t>> WebCryptoClient::wrapCryptoKey(const Vector<uint8_t>& key) const
{
    SEL selector = @selector(webCryptoMasterKeyForWebView:);
    Vector<uint8_t> wrappedKey;
    if ([[m_webView UIDelegate] respondsToSelector:selector]) {
        auto masterKey = makeVector(CallUIDelegate(m_webView, selector));
        if (!WebCore::wrapSerializedCryptoKey(masterKey, key, wrappedKey))
            return std::nullopt;
        return wrappedKey;
    }

    auto masterKey = WebCore::defaultWebCryptoMasterKey();
    if (!masterKey)
        return std::nullopt;
    if (!WebCore::wrapSerializedCryptoKey(WTFMove(*masterKey), key, wrappedKey))
        return std::nullopt;
    return wrappedKey;
}

std::optional<Vector<uint8_t>> WebCryptoClient::serializeAndWrapCryptoKey(WebCore::CryptoKeyData&& keyData) const
{
    auto key = WebCore::CryptoKey::create(WTFMove(keyData));
    if (!key)
        return std::nullopt;

    JSContextRef context = [[m_webView mainFrame] globalContext];
    auto serializedKey = WebCore::SerializedScriptValue::serializeCryptoKey(context, *key);
    return wrapCryptoKey(serializedKey);
}

std::optional<Vector<uint8_t>> WebCryptoClient::unwrapCryptoKey(const Vector<uint8_t>& serializedKey) const
{
    auto wrappedKey = WebCore::readSerializedCryptoKey(serializedKey);
    if (!wrappedKey)
        return std::nullopt;
    SEL selector = @selector(webCryptoMasterKeyForWebView:);
    if ([[m_webView UIDelegate] respondsToSelector:selector]) {
        auto masterKey = makeVector(CallUIDelegate(m_webView, selector));
        return WebCore::unwrapCryptoKey(masterKey, *wrappedKey);
    }
    if (auto masterKey = WebCore::defaultWebCryptoMasterKey())
        return WebCore::unwrapCryptoKey(*masterKey, *wrappedKey);
    return std::nullopt;
}

WebCryptoClient::WebCryptoClient(WebView* webView)
    : m_webView(webView)
{
}
