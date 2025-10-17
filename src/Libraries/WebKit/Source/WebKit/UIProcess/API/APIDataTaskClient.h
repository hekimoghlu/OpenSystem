/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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

#include "AuthenticationChallengeDisposition.h"
#include "AuthenticationChallengeProxy.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class Credential;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace API {

class Data;

class DataTaskClient : public RefCounted<DataTaskClient> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DataTaskClient);
public:
    static Ref<DataTaskClient> create() { return adoptRef(*new DataTaskClient); }
    virtual ~DataTaskClient() { }

    virtual void didReceiveChallenge(DataTask&, WebCore::AuthenticationChallenge&&, CompletionHandler<void(WebKit::AuthenticationChallengeDisposition, WebCore::Credential&&)>&& completionHandler) const { completionHandler(WebKit::AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue, { }); }
    virtual void willPerformHTTPRedirection(DataTask&, WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, CompletionHandler<void(bool)>&& completionHandler) const { completionHandler(true); }
    virtual void didReceiveResponse(DataTask&, WebCore::ResourceResponse&&, CompletionHandler<void(bool)>&& completionHandler) const { completionHandler(true); }
    virtual void didReceiveData(DataTask&, std::span<const uint8_t>) const { }
    virtual void didCompleteWithError(DataTask&, WebCore::ResourceError&&) const { }
};

} // namespace API
