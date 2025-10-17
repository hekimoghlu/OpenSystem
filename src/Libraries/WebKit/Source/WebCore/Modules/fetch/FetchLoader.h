/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

#include "ThreadableLoader.h"
#include "ThreadableLoaderClient.h"
#include "URLKeepingBlobAlive.h"
#include <wtf/CheckedPtr.h>
#include <wtf/URL.h>

namespace WebCore {

class Blob;
class FetchBodyConsumer;
class FetchLoaderClient;
class FetchRequest;
class ScriptExecutionContext;
class FragmentedSharedBuffer;

class WEBCORE_EXPORT FetchLoader final : public ThreadableLoaderClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(FetchLoader);
public:
    FetchLoader(FetchLoaderClient&, FetchBodyConsumer*);
    ~FetchLoader();

    RefPtr<FragmentedSharedBuffer> startStreaming();

    void start(ScriptExecutionContext&, const FetchRequest&, const String&);
    void start(ScriptExecutionContext&, const Blob&);
    void startLoadingBlobURL(ScriptExecutionContext&, const URL& blobURL);
    void stop();

    bool isStarted() const { return m_isStarted; }

private:
    // ThreadableLoaderClient API.
    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
    void didReceiveData(const SharedBuffer&) final;
    void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) final;
    void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) final;

private:
    CheckedRef<FetchLoaderClient> m_client;
    RefPtr<ThreadableLoader> m_loader;
    CheckedPtr<FetchBodyConsumer> m_consumer;
    bool m_isStarted { false };
    URLKeepingBlobAlive m_urlForReading;
};

} // namespace WebCore
