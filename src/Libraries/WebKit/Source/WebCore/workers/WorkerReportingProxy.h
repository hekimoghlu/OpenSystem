/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#include <wtf/CheckedPtr.h>

namespace WebCore {

class WorkerReportingProxy : public CanMakeThreadSafeCheckedPtr<WorkerReportingProxy> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerReportingProxy);
public:
    virtual ~WorkerReportingProxy() = default;

    virtual void postExceptionToWorkerObject(const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL) = 0;

    virtual void reportErrorToWorkerObject(const String&) = 0;

    // Invoked when close() is invoked on the worker context.
    virtual void workerGlobalScopeClosed() = 0;

    // Invoked when the thread has stopped.
    virtual void workerGlobalScopeDestroyed() = 0;

    // CanMakeCheckedPtr.
    virtual uint32_t checkedPtrCount() const = 0;
    virtual uint32_t checkedPtrCountWithoutThreadCheck() const = 0;
    virtual void incrementCheckedPtrCount() const = 0;
    virtual void decrementCheckedPtrCount() const = 0;
};

} // namespace WebCore
