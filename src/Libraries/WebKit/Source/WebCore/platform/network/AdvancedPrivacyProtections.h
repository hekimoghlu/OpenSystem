/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

namespace WebCore {

// FIXME: WebSearchContent is only used for debugging, and is unrelated to these other privacy protections;
// we should move it out of this enum and make it a separate property on per-navigation policies instead.
enum class AdvancedPrivacyProtections : uint16_t {
    BaselineProtections = 1 << 0,
    HTTPSFirst = 1 << 1,
    HTTPSOnly = 1 << 2,
    HTTPSOnlyExplicitlyBypassedForDomain = 1 << 3,
    FailClosedForUnreachableHosts = 1 << 4,
    WebSearchContent = 1 << 5,
    FingerprintingProtections = 1 << 6,
    EnhancedNetworkPrivacy = 1 << 7,
    LinkDecorationFiltering = 1 << 8,
    ScriptTelemetry = 1 << 9,
    FailClosedForAllHosts = 1 << 10,
};

}
