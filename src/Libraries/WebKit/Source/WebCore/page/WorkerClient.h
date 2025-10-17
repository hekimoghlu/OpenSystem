/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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

#include "GraphicsClient.h"
#include <wtf/FunctionDispatcher.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class WorkerClient : public GraphicsClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WorkerClient);
public:

    // Used for constructing clients for nested workers. Created on the worker thread of the outer worker, and then transferred to the nested worker.
    virtual UniqueRef<WorkerClient> createNestedWorkerClient(SerialFunctionDispatcher&) = 0;

    virtual ~WorkerClient() = default;
};

} // namespace WebCore
