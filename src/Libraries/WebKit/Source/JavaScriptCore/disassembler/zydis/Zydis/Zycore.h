/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
/**
 * @file
 * Master include file, including everything else.
 */

#ifndef ZYCORE_H
#define ZYCORE_H

#include "ZycoreExportConfig.h"
#include "ZycoreTypes.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

// TODO:

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================== */
/* Macros                                                                                         */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Constants                                                                                      */
/* ---------------------------------------------------------------------------------------------- */

/**
 * A macro that defines the zycore version.
 */
#define ZYCORE_VERSION (ZyanU64)0x0001000000000000

/* ---------------------------------------------------------------------------------------------- */
/* Helper macros                                                                                  */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Extracts the major-part of the zycore version.
 *
 * @param   version The zycore version value
 */
#define ZYCORE_VERSION_MAJOR(version) (ZyanU16)((version & 0xFFFF000000000000) >> 48)

/**
 * Extracts the minor-part of the zycore version.
 *
 * @param   version The zycore version value
 */
#define ZYCORE_VERSION_MINOR(version) (ZyanU16)((version & 0x0000FFFF00000000) >> 32)

/**
 * Extracts the patch-part of the zycore version.
 *
 * @param   version The zycore version value
 */
#define ZYCORE_VERSION_PATCH(version) (ZyanU16)((version & 0x00000000FFFF0000) >> 16)

/**
 * Extracts the build-part of the zycore version.
 *
 * @param   version The zycore version value
 */
#define ZYCORE_VERSION_BUILD(version) (ZyanU16)(version & 0x000000000000FFFF)

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

/**
 * Returns the zycore version.
 *
 * @return  The zycore version.
 *
 * Use the macros provided in this file to extract the major, minor, patch and build part from the
 * returned version value.
 */
ZYCORE_EXPORT ZyanU64 ZycoreGetVersion(void);

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYCORE_H */
