/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "CommonAtomStrings.h"

namespace WebCore {

#define DEFINE_COMMON_ATOM(atomName, atomValue) \
    MainThreadLazyNeverDestroyed<const AtomString> atomName ## AtomData;
#define INITIALIZE_COMMON_ATOM(atomName, atomValue) \
    atomName ## AtomData.constructWithoutAccessCheck(atomValue ## _s);

WEBCORE_COMMON_ATOM_STRINGS_FOR_EACH_KEYWORD(DEFINE_COMMON_ATOM)

void initializeCommonAtomStrings()
{
    // Initialization is not thread safe, so this function must be called from the main thread first.
    ASSERT(isUIThread());

    static std::once_flag initializeKey;
    std::call_once(initializeKey, [] {
        WEBCORE_COMMON_ATOM_STRINGS_FOR_EACH_KEYWORD(INITIALIZE_COMMON_ATOM)
    });
}

#undef DEFINE_COMMON_ATOM
#undef INITIALIZE_COMMON_ATOM

} // namespace WebCore
