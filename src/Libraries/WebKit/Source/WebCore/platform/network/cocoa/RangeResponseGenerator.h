/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/URLHash.h>

OBJC_CLASS NSURLRequest;
OBJC_CLASS WebCoreNSURLSessionDataTask;

namespace WebCore {

struct ParsedRequestRange;
class PlatformMediaResource;
class ResourceResponse;

class RangeResponseGenerator final
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RangeResponseGenerator, WTF::DestructionThread::Main> {
public:
    static Ref<RangeResponseGenerator> create(GuaranteedSerialFunctionDispatcher& dispatcher) { return adoptRef(*new RangeResponseGenerator(dispatcher)); }
    ~RangeResponseGenerator();

    bool willSynthesizeRangeResponses(WebCoreNSURLSessionDataTask *, PlatformMediaResource&, const ResourceResponse&);
    bool willHandleRequest(WebCoreNSURLSessionDataTask *, NSURLRequest *);
    void removeTask(WebCoreNSURLSessionDataTask *);

private:
    struct Data;

    RangeResponseGenerator(WTF::GuaranteedSerialFunctionDispatcher&);
    HashMap<String, std::unique_ptr<Data>>& map();

    class MediaResourceClient;
    void giveResponseToTasksWithFinishedRanges(Data&);
    void giveResponseToTaskIfBytesInRangeReceived(WebCoreNSURLSessionDataTask *, const ParsedRequestRange&, std::optional<size_t> expectedContentLength, const Data&);
    static std::optional<size_t> expectedContentLengthFromData(const Data&);

    HashMap<String, std::unique_ptr<Data>> m_map WTF_GUARDED_BY_CAPABILITY(m_targetDispatcher.get());
    Ref<GuaranteedSerialFunctionDispatcher> m_targetDispatcher;
};

} // namespace WebCore
