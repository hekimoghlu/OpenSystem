/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

#include <optional>
#include <wtf/Forward.h>

namespace JSC { namespace Yarr {

// Flags must be ordered in alphabet ordering.
#define JSC_REGEXP_FLAGS(macro) \
    macro('d', HasIndices, hasIndices, 0) \
    macro('g', Global, global, 1) \
    macro('i', IgnoreCase, ignoreCase, 2) \
    macro('m', Multiline, multiline, 3) \
    macro('s', DotAll, dotAll, 4) \
    macro('u', Unicode, unicode, 5) \
    macro('v', UnicodeSets, unicodeSets, 6) \
    macro('y', Sticky, sticky, 7) \

#define JSC_COUNT_REGEXP_FLAG(key, name, lowerCaseName, index) + 1
static constexpr unsigned numberOfFlags = 0 JSC_REGEXP_FLAGS(JSC_COUNT_REGEXP_FLAG);
#undef JSC_COUNT_REGEXP_FLAG

enum class Flags : uint16_t {
#define JSC_DEFINE_REGEXP_FLAG(key, name, lowerCaseName, index) name = 1 << index,
    JSC_REGEXP_FLAGS(JSC_DEFINE_REGEXP_FLAG)
#undef JSC_DEFINE_REGEXP_FLAG
    DeletedValue = 1 << numberOfFlags,
};

JS_EXPORT_PRIVATE std::optional<OptionSet<Flags>> parseFlags(StringView);
using FlagsString = std::array<char, Yarr::numberOfFlags + 1>; // numberOfFlags + null-terminator
JS_EXPORT_PRIVATE FlagsString flagsString(OptionSet<Flags>);

} } // namespace JSC::Yarr
