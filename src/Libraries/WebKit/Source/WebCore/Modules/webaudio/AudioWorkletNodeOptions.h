/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

#include "AudioNodeOptions.h"
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Strong.h>
#include <wtf/Forward.h>
#include <wtf/KeyValuePair.h>

namespace WebCore {

struct AudioWorkletNodeOptions : AudioNodeOptions {
    unsigned numberOfInputs = 1;
    unsigned numberOfOutputs = 1;
    std::optional<Vector<unsigned>> outputChannelCount;
    std::optional<Vector<KeyValuePair<String, double>>> parameterData;
    JSC::Strong<JSC::JSObject> processorOptions;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
