/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#include "TargetPolicy.h"
#include "Defines.h"

#if BUILDING_DYLD
    #if TARGET_OS_OSX
    const bool gHeaderAddImplicitPlatform = true;
    #else
    const bool gHeaderAddImplicitPlatform = false;
    #endif // TARGET_OS_OSX
#else
const bool gHeaderAddImplicitPlatform = false;
#endif // BUILDING_DYLD

#if BUILDING_LD || BUILDING_LD_UNIT_TESTS
const bool gHeaderAllowEmptyPlatform = true;
#else
const bool gHeaderAllowEmptyPlatform = false;
#endif // BUILDING_LD

#if BUILDING_LD || BUILDING_LD_UNIT_TESTS
// don't need deep inspection of dylibs we are linking with
const bool gImageValidateInitializers = false;
#else
const bool gImageValidateInitializers = true;
#endif

#if BUILDING_DYLD
// only dyld knows for sure that content was rebased when walking initializers
const bool gImageAssumeContentRebased = true;
#else
const bool gImageAssumeContentRebased = false;
#endif
