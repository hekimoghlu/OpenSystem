/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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

#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct WrappedCryptoKey;

// The purpose of the following APIs is to protect serialized CryptoKey data in IndexedDB or
// any other local storage that go through the structured clone algorithm. However, a side effect
// of this extra layer of protection is redundant communications between mainThread(document) and
// workerThreads. Please refer to WorkerGlobalScope for detailed explanation. P.S. This extra layer
// of protection is not required by the spec as of 11 December 2014:
// https://www.w3.org/TR/WebCryptoAPI/#security-developers

WEBCORE_EXPORT std::optional<Vector<uint8_t>> defaultWebCryptoMasterKey();
WEBCORE_EXPORT bool deleteDefaultWebCryptoMasterKey();

WEBCORE_EXPORT bool wrapSerializedCryptoKey(const Vector<uint8_t>& masterKey, const Vector<uint8_t>& key, Vector<uint8_t>& result);

WEBCORE_EXPORT std::optional<WrappedCryptoKey> readSerializedCryptoKey(const Vector<uint8_t>& wrappedKey);
WEBCORE_EXPORT std::optional<Vector<uint8_t>> unwrapCryptoKey(const Vector<uint8_t>& masterKey, const struct WrappedCryptoKey& wrappedKey);

} // namespace WebCore
