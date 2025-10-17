/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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

namespace WebKit {

ProcessID ProcessProviderLibWPE::launchProcess(const ProcessLauncher::LaunchOptions& launchOptions, char** argv, int childProcessSocket)
{
#if WPE_CHECK_VERSION(1, 14, 0)
    UNUSED_PARAM(childProcessSocket);
    if (!m_provider)
        return -1;

    if (wpe_process_launch(m_provider.get(), static_cast<wpe_process_type>(wpeProcessType(launchOptions.processType)), argv) > -1)
        return launchOptions.processIdentifier.toUInt64();
    return -1;
#else
    UNUSED_PARAM(launchOptions);
    UNUSED_PARAM(argv);
    UNUSED_PARAM(childProcessSocket);
    return -1;
#endif
}

} // namespace WebKit

#endif // USE(LIBWPE) && !ENABLE(BUBBLEWRAP_SANDBOX)
