/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#include "AbortController.h"

#include "AbortSignal.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AbortController);

Ref<AbortController> AbortController::create(ScriptExecutionContext& context)
{
    return adoptRef(*new AbortController(context));
}

AbortController::AbortController(ScriptExecutionContext& context)
    : m_signal(AbortSignal::create(&context))
{
}

AbortController::~AbortController() = default;

AbortSignal& AbortController::signal()
{
    return m_signal.get();
}

void AbortController::abort(JSC::JSValue reason)
{
    protectedSignal()->signalAbort(reason);
}

WebCoreOpaqueRoot AbortController::opaqueRoot()
{
    return root(&signal());
}

Ref<AbortSignal> AbortController::protectedSignal() const
{
    return m_signal;
}

}
