/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#include "AdvancedPrivacyProtections.h"
#include "Settings.h"
#include <JavaScriptCore/RuntimeFlags.h>
#include <wtf/URL.h>

namespace WebCore {

struct WorkletParameters {
    URL windowURL;
    JSC::RuntimeFlags jsRuntimeFlags;
    float sampleRate;
    String identifier;
    PAL::SessionID sessionID;
    Settings::Values settingsValues;
    ReferrerPolicy referrerPolicy;
    bool isAudioContextRealTime;
    OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections;
    std::optional<uint64_t> noiseInjectionHashSalt;

    WorkletParameters isolatedCopy() const & { return { windowURL.isolatedCopy(), jsRuntimeFlags, sampleRate, identifier.isolatedCopy(), sessionID, settingsValues.isolatedCopy(), referrerPolicy, isAudioContextRealTime, advancedPrivacyProtections, noiseInjectionHashSalt }; }
    WorkletParameters isolatedCopy() && { return { WTFMove(windowURL).isolatedCopy(), jsRuntimeFlags, sampleRate, WTFMove(identifier).isolatedCopy(), sessionID, WTFMove(settingsValues).isolatedCopy(), referrerPolicy, isAudioContextRealTime, advancedPrivacyProtections, WTFMove(noiseInjectionHashSalt) }; }
};

} // namespace WebCore
