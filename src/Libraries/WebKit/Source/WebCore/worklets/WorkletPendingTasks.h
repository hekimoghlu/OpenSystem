/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#include "JSDOMPromiseDeferred.h"
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Worklet;

// https://drafts.css-houdini.org/worklets/#pending-tasks-struct
class WorkletPendingTasks : public ThreadSafeRefCounted<WorkletPendingTasks> {
public:
    static Ref<WorkletPendingTasks> create(Worklet& worklet, DOMPromiseDeferred<void>&& promise, int counter)
    {
        return adoptRef(*new WorkletPendingTasks(worklet, WTFMove(promise), counter));
    }

    void abort(Exception&&);
    void decrementCounter();

private:
    WorkletPendingTasks(Worklet&, DOMPromiseDeferred<void>&&, int counter);

    WeakPtr<Worklet> m_worklet;
    DOMPromiseDeferred<void> m_promise;
    int m_counter;
};

} // namespace WebCore
