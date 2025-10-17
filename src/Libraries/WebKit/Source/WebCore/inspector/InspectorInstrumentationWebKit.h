/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#include "InspectorInstrumentationPublic.h"
#include "ResourceLoader.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Function.h>

namespace WebCore {

class LocalFrame;
class ResourceLoader;
class ResourceRequest;
class ResourceResponse;
class FragmentedSharedBuffer;

class WEBCORE_EXPORT InspectorInstrumentationWebKit {
public:
    static bool shouldInterceptRequest(const ResourceLoader&);
    static bool shouldInterceptResponse(const LocalFrame*, const ResourceResponse&);
    static void interceptRequest(ResourceLoader&, Function<void(const ResourceRequest&)>&&);
    static void interceptResponse(const LocalFrame*, const ResourceResponse&, ResourceLoaderIdentifier, CompletionHandler<void(const ResourceResponse&, RefPtr<FragmentedSharedBuffer>)>&&);

private:
    static bool shouldInterceptRequestInternal(const ResourceLoader&);
    static bool shouldInterceptResponseInternal(const LocalFrame&, const ResourceResponse&);
    static void interceptRequestInternal(ResourceLoader&, Function<void(const ResourceRequest&)>&&);
    static void interceptResponseInternal(const LocalFrame&, const ResourceResponse&, ResourceLoaderIdentifier, CompletionHandler<void(const ResourceResponse&, RefPtr<FragmentedSharedBuffer>)>&&);
};

inline bool InspectorInstrumentationWebKit::shouldInterceptRequest(const ResourceLoader& loader)
{
    FAST_RETURN_IF_NO_FRONTENDS(false);
    return shouldInterceptRequestInternal(loader);
}

inline bool InspectorInstrumentationWebKit::shouldInterceptResponse(const LocalFrame* frame, const ResourceResponse& response)
{
    FAST_RETURN_IF_NO_FRONTENDS(false);
    if (!frame)
        return false;

    return shouldInterceptResponseInternal(*frame, response);
}

inline void InspectorInstrumentationWebKit::interceptRequest(ResourceLoader& loader, Function<void(const ResourceRequest&)>&& handler)
{
    ASSERT(InspectorInstrumentationWebKit::shouldInterceptRequest(loader));
    interceptRequestInternal(loader, WTFMove(handler));
}

inline void InspectorInstrumentationWebKit::interceptResponse(const LocalFrame* frame, const ResourceResponse& response, ResourceLoaderIdentifier identifier, CompletionHandler<void(const ResourceResponse&, RefPtr<FragmentedSharedBuffer>)>&& handler)
{
    ASSERT(InspectorInstrumentationWebKit::shouldInterceptResponse(frame, response));
    interceptResponseInternal(*frame, response, identifier, WTFMove(handler));
}

}
