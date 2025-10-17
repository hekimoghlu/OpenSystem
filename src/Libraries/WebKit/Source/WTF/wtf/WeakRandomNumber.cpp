/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#include <wtf/WeakRandomNumber.h>

#include <wtf/WeakRandom.h>

namespace WTF {

template<> unsigned weakRandomNumber<unsigned>()
{
    // We don't care about thread safety. WeakRandom just uses POD types,
    // and racy access just increases randomness.
    static WeakRandom s_weakRandom;
    return s_weakRandom.getUint32();
}

#if PLATFORM(COCOA)

// Older versions of Safari frameworks were using weakRandomUint32.
// Remove this once we no longer need to support those.

WTF_EXPORT_PRIVATE unsigned weakRandomUint32();

unsigned weakRandomUint32()
{
    return random();
}

#endif

}
