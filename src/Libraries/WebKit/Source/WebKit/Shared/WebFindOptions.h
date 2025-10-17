/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

#include <WebCore/FindOptions.h>

namespace WebKit {

enum class FindOptions : uint16_t {
    CaseInsensitive = 1 << 0,
    AtWordStarts = 1 << 1,
    TreatMedialCapitalAsWordStart = 1 << 2,
    Backwards = 1 << 3,
    WrapAround = 1 << 4,
    ShowOverlay = 1 << 5,
    ShowFindIndicator = 1 << 6,
    ShowHighlight = 1 << 7,
    DetermineMatchIndex = 1 << 8,
    NoIndexChange = 1 << 9,
    AtWordEnds = 1 << 10,
    DoNotSetSelection = 1 << 11,
};

enum class FindDecorationStyle : uint8_t {
    Normal,
    Found,
    Highlighted,
};

WebCore::FindOptions core(OptionSet<FindOptions>);

} // namespace WebKit
