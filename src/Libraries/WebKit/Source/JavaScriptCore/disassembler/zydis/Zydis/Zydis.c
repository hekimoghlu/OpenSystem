/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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

#if ENABLE(ZYDIS)

#include "Zydis.h"

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

ZyanU64 ZydisGetVersion(void)
{
    return ZYDIS_VERSION;
}

ZyanStatus ZydisIsFeatureEnabled(ZydisFeature feature)
{
    switch (feature)
    {
    case ZYDIS_FEATURE_DECODER:
#ifndef ZYDIS_DISABLE_DECODER
        return ZYAN_STATUS_TRUE;
#else
        return ZYAN_STATUS_FALSE;
#endif
    case ZYDIS_FEATURE_FORMATTER:
#ifndef ZYDIS_DISABLE_FORMATTER
        return ZYAN_STATUS_TRUE;
#else
        return ZYAN_STATUS_FALSE;
#endif
    case ZYDIS_FEATURE_AVX512:
#ifndef ZYDIS_DISABLE_AVX512
        return ZYAN_STATUS_TRUE;
#else
        return ZYAN_STATUS_FALSE;
#endif

    case ZYDIS_FEATURE_KNC:
#ifndef ZYDIS_DISABLE_KNC
        return ZYAN_STATUS_TRUE;
#else
        return ZYAN_STATUS_FALSE;
#endif

    default:
        return ZYAN_STATUS_INVALID_ARGUMENT;
    }
}

/* ============================================================================================== */

#endif /* ENABLE(ZYDIS) */
