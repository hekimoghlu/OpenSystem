/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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

#include "EventTarget.h"
#include "ExceptionOr.h"
#include "FetchOptions.h"

namespace WebCore {

struct FetchOptions;
struct WorkerOptions;

class AbstractWorker : public RefCounted<AbstractWorker>, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AbstractWorker);
public:
    using RefCounted::ref;
    using RefCounted::deref;

    static FetchOptions workerFetchOptions(const WorkerOptions&, FetchOptions::Destination);

protected:
    AbstractWorker() = default;

    // Helper function that converts a URL to an absolute URL and checks the result for validity.
    ExceptionOr<URL> resolveURL(const String& url);

    intptr_t asID() const { return reinterpret_cast<intptr_t>(this); }

private:
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
};

} // namespace WebCore
