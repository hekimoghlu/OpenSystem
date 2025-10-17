/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#if ENABLE(WEB_AUDIO)

#include "AutomationRate.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

struct AudioParamDescriptor {
    String name;
    float defaultValue { 0 };
    float minValue { -3.4028235e38 };
    float maxValue { 3.4028235e38 };
    AutomationRate automationRate { AutomationRate::ARate };

    AudioParamDescriptor isolatedCopy() const & { return { name.isolatedCopy(), defaultValue, minValue, maxValue, automationRate }; }
    AudioParamDescriptor isolatedCopy() && { return { WTFMove(name).isolatedCopy(), defaultValue, minValue, maxValue, automationRate }; }
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
