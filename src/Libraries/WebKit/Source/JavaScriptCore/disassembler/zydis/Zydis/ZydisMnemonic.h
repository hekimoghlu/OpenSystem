/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
 * Mnemonic constant definitions and helper functions.
 */

#ifndef ZYDIS_MNEMONIC_H
#define ZYDIS_MNEMONIC_H

#include "ZycoreTypes.h"
#include "ZydisShortString.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================== */
/* Enums and types                                                                                */
/* ============================================================================================== */

#include "ZydisGeneratedEnumMnemonic.h"

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

/**
 * @addtogroup mnemonic Mnemonic
 * Functions for retrieving mnemonic names.
 * @{
 */

/**
 * Returns the specified instruction mnemonic string.
 *
 * @param   mnemonic    The mnemonic.
 *
 * @return  The instruction mnemonic string or `ZYAN_NULL`, if an invalid mnemonic was passed.
 */
ZYDIS_EXPORT const char* ZydisMnemonicGetString(ZydisMnemonic mnemonic);

/**
 * Returns the specified instruction mnemonic as `ZydisShortString`.
 *
 * @param   mnemonic    The mnemonic.
 *
 * @return  The instruction mnemonic string or `ZYAN_NULL`, if an invalid mnemonic was passed.
 *
 * The `buffer` of the returned struct is guaranteed to be zero-terminated in this special case.
 */
ZYDIS_EXPORT const ZydisShortString* ZydisMnemonicGetStringWrapped(ZydisMnemonic mnemonic);

/**
 * @}
 */

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYDIS_MNEMONIC_H */
