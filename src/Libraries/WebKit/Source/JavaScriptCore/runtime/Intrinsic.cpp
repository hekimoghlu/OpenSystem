/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#include "config.h"
#include "Intrinsic.h"

#include <wtf/PrintStream.h>

namespace JSC {

ASCIILiteral intrinsicName(Intrinsic intrinsic)
{
    switch (intrinsic) {
#define JSC_INTRINSIC_STRING(name) case name: return #name ""_s;
    JSC_FOR_EACH_INTRINSIC(JSC_INTRINSIC_STRING)
#undef JSC_INTRINSIC_STRING
    }
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}

std::optional<IterationKind> interationKindForIntrinsic(Intrinsic intrinsic)
{
    switch (intrinsic) {
    case ArrayValuesIntrinsic:
    case TypedArrayValuesIntrinsic:
        return IterationKind::Values;
    case ArrayKeysIntrinsic:
    case TypedArrayKeysIntrinsic:
        return IterationKind::Keys;
    case ArrayEntriesIntrinsic:
    case TypedArrayEntriesIntrinsic:
        return IterationKind::Entries;
    default:
        return std::nullopt;
    }
}


} // namespace JSC

namespace WTF {

void printInternal(PrintStream& out, JSC::Intrinsic intrinsic)
{
    out.print(JSC::intrinsicName(intrinsic));
}

} // namespace WTF

