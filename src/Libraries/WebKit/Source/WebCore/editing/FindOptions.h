/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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

#include <wtf/OptionSet.h>

namespace WebCore {

enum class FindOption : uint16_t {
    CaseInsensitive = 1 << 0,
    AtWordStarts = 1 << 1,
    // When combined with AtWordStarts, accepts a match in the middle of a word if the match begins with
    // an uppercase letter followed by a lowercase or non-letter. Accepts several other intra-word matches.
    TreatMedialCapitalAsWordStart = 1 << 2,
    Backwards = 1 << 3,
    WrapAround = 1 << 4,
    StartInSelection = 1 << 5,
    DoNotRevealSelection = 1 << 6,
    AtWordEnds = 1 << 7,
    DoNotTraverseFlatTree = 1 << 8,
    DoNotSetSelection = 1 << 9,
};

using FindOptions = OptionSet<FindOption>;

} // namespace WebCore
