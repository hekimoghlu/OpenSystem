/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
#include "ProcessProviderLibWPE.h"

#if USE(LIBWPE) && !ENABLE(BUBBLEWRAP_SANDBOX)

#include <wpe/wpe.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

int ProcessProviderLibWPE::wpeProcessType(ProcessLauncher::ProcessType processType)
{
#if WPE_CHECK_VERSION(1, 14, 0)
    switch (processType) {
    case ProcessLauncher::ProcessType::Network:
        return WPE_PROCESS_TYPE_NETWORK;
#if ENABLE(GPU_PROCESS)
    case ProcessLauncher::ProcessType::GPU:
        return WPE_PROCESS_TYPE_GPU;
#endif
    case ProcessLauncher::ProcessType::Web:
    default:
        return WPE_PROCESS_TYPE_WEB;
    }
#else
    UNUSED_PARAM(processType);
    ASSERT_NOT_REACHED();
    return -1;
#endif
}

ProcessProviderLibWPE& ProcessProviderLibWPE::singleton()
{
    static NeverDestroyed<ProcessProviderLibWPE> sharedProvider;
    return sharedProvider;
}

ProcessProviderLibWPE::ProcessProviderLibWPE()
#if WPE_CHECK_VERSION(1, 14, 0)
    : m_provider(wpe_process_provider_create(), wpe_process_provider_destroy)
#else
    : m_provider(nullptr, nullptr)
#endif
{
}

bool ProcessProviderLibWPE::isEnabled()
{
    return m_provider.get();
}

void ProcessProviderLibWPE::kill(ProcessID processID)
{
#if WPE_CHECK_VERSION(1, 14, 0)
    if (!m_provider)
        return;

    wpe_process_terminate(m_provider.get(), processID);
#else
    UNUSED_PARAM(processID);
#endif
}

} // namespace WebKit

#endif // USE(LIBWPE) && !ENABLE(BUBBLEWRAP_SANDBOX)
