/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
#include "CacheStorageConnection.h"

#include "FetchResponse.h"
#include <wtf/CryptographicallyRandomNumber.h>

namespace WebCore {

using namespace WebCore::DOMCacheEngine;

static inline uint64_t formDataSize(const FormData& formData)
{
    if (isMainThread())
        return formData.lengthInBytes();

    uint64_t resultSize;
    callOnMainThreadAndWait([formData = formData.isolatedCopy(), &resultSize] {
        resultSize = formData->lengthInBytes();
    });
    return resultSize;
}

uint64_t CacheStorageConnection::computeRealBodySize(const DOMCacheEngine::ResponseBody& body)
{
    uint64_t result = 0;
    WTF::switchOn(body, [&] (const Ref<FormData>& formData) {
        result = formDataSize(formData);
    }, [&] (const Ref<SharedBuffer>& buffer) {
        result = buffer->size();
    }, [] (const std::nullptr_t&) {
    });
    return result;
}

uint64_t CacheStorageConnection::computeRecordBodySize(const FetchResponse& response, const DOMCacheEngine::ResponseBody& body)
{
    if (!response.opaqueLoadIdentifier()) {
        ASSERT(response.tainting() != ResourceResponse::Tainting::Opaque);
        return computeRealBodySize(body);
    }

    return m_opaqueResponseToSizeWithPaddingMap.ensure(response.opaqueLoadIdentifier(), [&] () {
        uint64_t realSize = computeRealBodySize(body);

        // Padding the size as per https://github.com/whatwg/storage/issues/31.
        uint64_t sizeWithPadding = realSize + static_cast<uint64_t>(cryptographicallyRandomUnitInterval() * 128000);
        sizeWithPadding = ((sizeWithPadding / 32000) + 1) * 32000;

        m_opaqueResponseToSizeWithPaddingMap.set(response.opaqueLoadIdentifier(), sizeWithPadding);
        return sizeWithPadding;
    }).iterator->value;
}

} // namespace WebCore
