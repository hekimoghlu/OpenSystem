/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#import "config.h"
#import "ModelProcess.h"

#if ENABLE(MODEL_PROCESS)

#import "ModelConnectionToWebProcess.h"
#import <wtf/RetainPtr.h>

namespace WebKit {
using namespace WebCore;

#if ENABLE(CFPREFS_DIRECT_MODE)
void ModelProcess::dispatchSimulatedNotificationsForPreferenceChange(const String& key)
{
}
#endif // ENABLE(CFPREFS_DIRECT_MODE)

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
