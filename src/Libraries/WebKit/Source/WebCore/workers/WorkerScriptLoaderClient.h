/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include "ResourceLoaderIdentifier.h"
#include "ScriptExecutionContextIdentifier.h"

namespace WebCore {
class WorkerScriptLoaderClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::WorkerScriptLoaderClient> : std::true_type { };
}

namespace WebCore {

class ResourceResponse;

class WorkerScriptLoaderClient : public CanMakeWeakPtr<WorkerScriptLoaderClient> {
public:
    virtual void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) = 0;
    virtual void notifyFinished(std::optional<ScriptExecutionContextIdentifier>) = 0;

protected:
    virtual ~WorkerScriptLoaderClient() = default;
};

} // namespace WebCore
