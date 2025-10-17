/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include "VMEntryScope.h"

#include "Options.h"
#include "SamplingProfiler.h"
#include "VM.h"
#include "VMEntryScopeInlines.h"
#include "WasmCapabilities.h"
#include "WasmMachineThreads.h"
#include "Watchdog.h"
#include <wtf/WTFConfig.h>

namespace JSC {

void VMEntryScope::setUpSlow()
{
    m_vm.entryScope = this;

    auto& thread = Thread::current();
    if (UNLIKELY(!thread.isJSThread())) {
        Thread::registerJSThread(thread);

        if (Wasm::isSupported())
            Wasm::startTrackingCurrentThread();
#if HAVE(MACH_EXCEPTIONS)
        registerThreadForMachExceptionHandling(thread);
#endif
    }

    if (UNLIKELY(m_vm.hasAnyEntryScopeServiceRequest() || m_vm.hasTimeZoneChange()))
        m_vm.executeEntryScopeServicesOnEntry();
}

void VMEntryScope::tearDownSlow()
{
    ASSERT_WITH_MESSAGE(!m_vm.hasCheckpointOSRSideState(), "Exitting the VM but pending checkpoint side state still available");

    m_vm.entryScope = nullptr;

    if (UNLIKELY(m_vm.hasAnyEntryScopeServiceRequest()))
        m_vm.executeEntryScopeServicesOnExit();
}

} // namespace JSC
