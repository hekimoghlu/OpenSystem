/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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

#include "JSDOMPromiseDeferredForward.h"
#include "NavigatorBase.h"
#include "Supplementable.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPU;

class WorkerNavigator final : public NavigatorBase, public Supplementable<WorkerNavigator> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WorkerNavigator);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerNavigator);
public:
    static Ref<WorkerNavigator> create(ScriptExecutionContext& context, const String& userAgent, bool isOnline) { return adoptRef(*new WorkerNavigator(context, userAgent, isOnline)); }

    virtual ~WorkerNavigator();

    const String& userAgent() const final;
    bool onLine() const final;
    void setIsOnline(bool isOnline) { m_isOnline = isOnline; }

    void setAppBadge(std::optional<unsigned long long>, Ref<DeferredPromise>&&);
    void clearAppBadge(Ref<DeferredPromise>&&);

    GPU* gpu();

private:
    explicit WorkerNavigator(ScriptExecutionContext&, const String&, bool isOnline);

    String m_userAgent;
    bool m_isOnline;
#if HAVE(WEBGPU_IMPLEMENTATION)
    RefPtr<GPU> m_gpuForWebGPU;
#endif
};

} // namespace WebCore
