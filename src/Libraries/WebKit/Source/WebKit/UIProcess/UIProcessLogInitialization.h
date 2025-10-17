/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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

#include <wtf/Forward.h>

// NOTE: These methods configure logging separately and differently for the UIProcess
// than similar methods elsewhere. Specifically, the log strings returned from these
// methods are initialized once, rather than repeatedly, and are intended to be passed
// through to AuxiliaryProcesses to further reduce the runtime cost of log string
// initialization.

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

namespace WebKit {

namespace UIProcess {

void initializeLoggingIfNecessary();
String wtfLogLevelString();
String webCoreLogLevelString();
String webKitLogLevelString();

}

}

#endif
