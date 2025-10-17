/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include "InspectorInstrumentationWebKit.h"

#include "InspectorInstrumentation.h"

namespace WebCore {

bool InspectorInstrumentationWebKit::shouldInterceptRequestInternal(const ResourceLoader& loader)
{
    return InspectorInstrumentation::shouldInterceptRequest(loader);
}

bool InspectorInstrumentationWebKit::shouldInterceptResponseInternal(const LocalFrame& frame, const ResourceResponse& response)
{
    return InspectorInstrumentation::shouldInterceptResponse(frame, response);
}

void InspectorInstrumentationWebKit::interceptRequestInternal(ResourceLoader& loader, Function<void(const ResourceRequest&)>&& handler)
{
    InspectorInstrumentation::interceptRequest(loader, WTFMove(handler));
}

void InspectorInstrumentationWebKit::interceptResponseInternal(const LocalFrame& frame, const ResourceResponse& response, ResourceLoaderIdentifier identifier, CompletionHandler<void(const ResourceResponse&, RefPtr<FragmentedSharedBuffer>)>&& handler)
{
    InspectorInstrumentation::interceptResponse(frame, response, identifier, WTFMove(handler));
}

} // namespace WebCore
