/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#include "DOMPromiseProxy.h"
#include <wtf/UniqueRef.h>

namespace WebCore {

struct BackgroundFetchRecordInformation;
class FetchRequest;
class FetchResponse;
class ScriptExecutionContext;

class BackgroundFetchRecord : public RefCounted<BackgroundFetchRecord> {
public:
    static Ref<BackgroundFetchRecord> create(ScriptExecutionContext& context, BackgroundFetchRecordInformation&& information) { return adoptRef(*new BackgroundFetchRecord(context, WTFMove(information))); }

    ~BackgroundFetchRecord();
    
    using ResponseReadyPromise = DOMPromiseProxy<IDLInterface<FetchResponse>>;
    ResponseReadyPromise& responseReady() { return m_responseReadyPromise; }
    Ref<FetchRequest> request();

    void settleResponseReadyPromise(ExceptionOr<Ref<FetchResponse>>&&);

private:
    BackgroundFetchRecord(ScriptExecutionContext&, BackgroundFetchRecordInformation&&);

    UniqueRef<ResponseReadyPromise> m_responseReadyPromise;
    Ref<FetchRequest> m_request;
};

} // namespace WebCore
