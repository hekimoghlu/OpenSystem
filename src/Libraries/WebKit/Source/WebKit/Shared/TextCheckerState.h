/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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

namespace WebKit {

enum class TextCheckerState : uint8_t {
    ContinuousSpellCheckingEnabled = 1 << 0,
    GrammarCheckingEnabled = 1 << 1,
#if USE(APPKIT)
    AutomaticTextReplacementEnabled = 1 << 2,
    AutomaticSpellingCorrectionEnabled = 1 << 3,
    AutomaticQuoteSubstitutionEnabled = 1 << 4,
    AutomaticDashSubstitutionEnabled = 1 << 5,
    AutomaticLinkDetectionEnabled = 1 << 6,
#endif
};

} // namespace WebKit
