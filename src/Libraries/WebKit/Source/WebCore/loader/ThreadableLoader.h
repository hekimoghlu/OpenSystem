/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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

#include "ResourceLoaderOptions.h"
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

    class ResourceError;
    class ResourceRequest;
    class ResourceResponse;
    class ScriptExecutionContext;
    class ThreadableLoaderClient;

    enum class ContentSecurityPolicyEnforcement {
        DoNotEnforce,
        EnforceWorkerSrcDirective,
        EnforceConnectSrcDirective,
        EnforceScriptSrcDirective,
    };

    enum class ResponseFilteringPolicy {
        Enable,
        Disable,
    };

    struct ThreadableLoaderOptions : ResourceLoaderOptions {
        ThreadableLoaderOptions();
        explicit ThreadableLoaderOptions(FetchOptions&&);
        ThreadableLoaderOptions(const ResourceLoaderOptions&, ContentSecurityPolicyEnforcement, String&& initiatorType, ResponseFilteringPolicy);
        ~ThreadableLoaderOptions();

        ThreadableLoaderOptions isolatedCopy() const;

        ContentSecurityPolicyEnforcement contentSecurityPolicyEnforcement { ContentSecurityPolicyEnforcement::EnforceConnectSrcDirective };
        String initiatorType; // This cannot be an AtomString, as isolatedCopy() wouldn't create an object that's safe for passing to another thread.
        ResponseFilteringPolicy filteringPolicy { ResponseFilteringPolicy::Disable };
    };

    // Useful for doing loader operations from any thread (not threadsafe,
    // just able to run on threads other than the main thread).
    class ThreadableLoader {
        WTF_MAKE_NONCOPYABLE(ThreadableLoader);
    public:
        static void loadResourceSynchronously(ScriptExecutionContext&, ResourceRequest&&, ThreadableLoaderClient&, const ThreadableLoaderOptions&);
        static RefPtr<ThreadableLoader> create(ScriptExecutionContext&, ThreadableLoaderClient&, ResourceRequest&&, const ThreadableLoaderOptions&, String&& referrer = String(), String&& taskMode = { });

        virtual void computeIsDone() = 0;
        virtual void cancel() = 0;
        void ref() { refThreadableLoader(); }
        void deref() { derefThreadableLoader(); }

        static void logError(ScriptExecutionContext&, const ResourceError&, const String&);

    protected:
        ThreadableLoader() = default;
        virtual ~ThreadableLoader() = default;
        virtual void refThreadableLoader() = 0;
        virtual void derefThreadableLoader() = 0;
    };

} // namespace WebCore
