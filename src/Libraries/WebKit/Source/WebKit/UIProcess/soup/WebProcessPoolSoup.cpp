/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "WebProcessPool.h"

#include "MemoryPressureMonitor.h"
#include "NetworkProcessCreationParameters.h"
#include "NetworkProcessMessages.h"
#include "OverrideLanguages.h"
#include "WebProcessMessages.h"
#include <wtf/Language.h>

namespace WebKit {

std::optional<MemoryPressureHandler::Configuration> WebProcessPool::s_networkProcessMemoryPressureHandlerConfiguration;

void WebProcessPool::platformInitializeNetworkProcess(NetworkProcessCreationParameters& parameters)
{
    parameters.languages = overrideLanguages().isEmpty() ? userPreferredLanguages() : overrideLanguages();
    parameters.memoryPressureHandlerConfiguration = s_networkProcessMemoryPressureHandlerConfiguration;

#if OS(LINUX)
    if (MemoryPressureMonitor::disabled())
        parameters.shouldSuppressMemoryPressureHandler = true;
#endif // OS(LINUX)
}

} // namespace WebKit
