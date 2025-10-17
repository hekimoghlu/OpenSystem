/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
 * Defines the immutable and storage-efficient `ZydisShortString` struct, which
 * is used to store strings in the generated tables.
 */

#ifndef ZYDIS_SHORTSTRING_H
#define ZYDIS_SHORTSTRING_H

#include "ZydisExportConfig.h"
#include "ZycoreDefines.h"
#include "ZycoreTypes.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================== */
/* Enums and types                                                                                */
/* ============================================================================================== */

/**
 * Defines the `ZydisShortString` struct.
 *
 * This compact struct is mainly used for internal string-tables to save up some bytes.
 *
 * All fields in this struct should be considered as "private". Any changes may lead to unexpected
 * behavior.
 */
typedef struct ZydisShortString_
{
    /**
     * The buffer that contains the actual (null-terminated) string.
    */
    const char* data;
    /**
     * The length (number of characters) of the string (without 0-termination).
    */
    ZyanU8 size;
} ZydisShortString;

/* ============================================================================================== */
/* Macros                                                                                         */
/* ============================================================================================== */

/**
 * Declares a `ZydisShortString` from a static C-style string.
 *
 * @param   string  The C-string constant.
 */
#define ZYDIS_MAKE_SHORTSTRING(string) \
    { string, sizeof(string) - 1 }

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYDIS_SHORTSTRING_H */
