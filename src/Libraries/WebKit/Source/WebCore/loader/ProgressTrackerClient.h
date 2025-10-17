/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

namespace WebCore {

class LocalFrame;

class ProgressTrackerClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    virtual ~ProgressTrackerClient() = default;

    virtual void willChangeEstimatedProgress() { }
    virtual void didChangeEstimatedProgress() { }

    virtual void progressStarted(LocalFrame& originatingProgressFrame) = 0;
    virtual void progressEstimateChanged(LocalFrame& originatingProgressFrame) = 0;
    virtual void progressFinished(LocalFrame& originatingProgressFrame) = 0;
};

} // namespace WebCore
