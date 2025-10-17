/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#include "ResourceError.h"

#if USE(CURL)

#include "CurlContext.h"

namespace WebCore {

static const String& curlErrorDomain()
{
    static NeverDestroyed<const String> errorDomain(MAKE_STATIC_STRING_IMPL("CurlErrorDomain"));
    return errorDomain;
}

ResourceError::ResourceError(int curlCode, const URL& failingURL, Type type)
    : ResourceErrorBase(curlErrorDomain(), curlCode, failingURL, CurlHandle::errorDescription(static_cast<CURLcode>(curlCode)), type, IsSanitized::No)
{
}

ResourceError ResourceError::fromIPCData(std::optional<IPCData>&& ipcData)
{
    if (!ipcData)
        return { };

    return {
        ipcData->domain,
        ipcData->errorCode,
        ipcData->failingURL,
        ipcData->localizedDescription,
        ipcData->type,
        ipcData->isSanitized
    };
}

auto ResourceError::ipcData() const -> std::optional<IPCData>
{
    if (isNull())
        return std::nullopt;

    return IPCData {
        type(),
        domain(),
        errorCode(),
        failingURL(),
        localizedDescription(),
        m_isSanitized
    };
}

bool ResourceError::isCertificationVerificationError() const
{
    return domain() == curlErrorDomain() && errorCode() == CURLE_PEER_FAILED_VERIFICATION;
}

void ResourceError::doPlatformIsolatedCopy(const ResourceError&)
{
}

bool ResourceError::platformCompare(const ResourceError&, const ResourceError&)
{
    return true;
}

} // namespace WebCore

#endif
