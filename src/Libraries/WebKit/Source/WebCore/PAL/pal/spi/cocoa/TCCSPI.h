/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

#if USE(APPLE_INTERNAL_SDK)

#import <TCC/TCC.h>

#else

#include <os/object.h>

typedef enum {
    kTCCAccessPreflightGranted,
    kTCCAccessPreflightDenied,
} TCCAccessPreflightResult;

#if HAVE(TCC_IOS_14_BIG_SUR_SPI)
typedef uint64_t tcc_identity_type_t;
constexpr tcc_identity_type_t TCC_IDENTITY_CODE_BUNDLE_ID = 0;
OS_OBJECT_DECL_CLASS(tcc_identity);
#endif // HAVE(TCC_IOS_14_BIG_SUR_SPI)

#endif
