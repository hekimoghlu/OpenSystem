/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#include "LoaderMalloc.h"
#include "ResourceLoaderIdentifier.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/CheckedRef.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class NetworkLoadMetrics;
class ResourceError;
class ResourceResponse;
class ResourceTiming;
class SharedBuffer;

class ThreadableLoaderClient : public CanMakeWeakPtr<ThreadableLoaderClient>, public CanMakeThreadSafeCheckedPtr<ThreadableLoaderClient> {
    WTF_MAKE_NONCOPYABLE(ThreadableLoaderClient);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ThreadableLoaderClient);
public:
    virtual void didSendData(unsigned long long /*bytesSent*/, unsigned long long /*totalBytesToBeSent*/) { }

    virtual void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) { }
    virtual void didReceiveData(const SharedBuffer&) { }
    virtual void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) { }
    virtual void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) { }
    virtual void didFinishTiming(const ResourceTiming&) { }
    virtual void notifyIsDone(bool) { ASSERT_NOT_REACHED(); }

protected:
    ThreadableLoaderClient() = default;
    virtual ~ThreadableLoaderClient() = default;
};

} // namespace WebCore
