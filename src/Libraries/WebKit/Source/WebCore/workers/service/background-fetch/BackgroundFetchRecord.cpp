/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include "BackgroundFetchRecord.h"

#include "BackgroundFetchRecordInformation.h"
#include "FetchRequest.h"
#include "JSFetchResponse.h"

namespace WebCore {

BackgroundFetchRecord::BackgroundFetchRecord(ScriptExecutionContext& context, BackgroundFetchRecordInformation&& information)
    : m_responseReadyPromise(makeUniqueRef<ResponseReadyPromise>())
    , m_request(FetchRequest::create(context, { }, FetchHeaders::create(information.guard, WTFMove(information.httpHeaders)), WTFMove(information.internalRequest), WTFMove(information.options), WTFMove(information.referrer)))
{
    // FIXME: We should provide a body to the request.
}

BackgroundFetchRecord::~BackgroundFetchRecord()
{
}

Ref<FetchRequest> BackgroundFetchRecord::request()
{
    return m_request;
}

void BackgroundFetchRecord::settleResponseReadyPromise(ExceptionOr<Ref<FetchResponse>>&& result)
{
    if (result.hasException()) {
        m_responseReadyPromise->reject(result.releaseException());
        return;
    }
    m_responseReadyPromise->resolveWithNewlyCreated(result.releaseReturnValue());
}

} // namespace WebCore
