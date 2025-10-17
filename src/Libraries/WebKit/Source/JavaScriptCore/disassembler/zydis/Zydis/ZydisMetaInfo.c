/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

#include "ZydisMetaInfo.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

/* ============================================================================================== */
/* Enum strings                                                                                   */
/* ============================================================================================== */

#include "ZydisGeneratedEnumInstructionCategory.inc"
#include "ZydisGeneratedEnumISASet.inc"
#include "ZydisGeneratedEnumISAExt.inc"

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

const char* ZydisCategoryGetString(ZydisInstructionCategory category)
{
    if ((ZyanUSize)category >= ZYAN_ARRAY_LENGTH(STR_INSTRUCTIONCATEGORY))
    {
        return ZYAN_NULL;
    }
    return STR_INSTRUCTIONCATEGORY[category];
}

const char* ZydisISASetGetString(ZydisISASet isa_set)
{
    if ((ZyanUSize)isa_set >= ZYAN_ARRAY_LENGTH(STR_ISASET))
    {
        return ZYAN_NULL;
    }
    return STR_ISASET[isa_set];
}

const char* ZydisISAExtGetString(ZydisISAExt isa_ext)
{
    if ((ZyanUSize)isa_ext >= ZYAN_ARRAY_LENGTH(STR_ISAEXT))
    {
        return ZYAN_NULL;
    }
    return STR_ISAEXT[isa_ext];
}

/* ============================================================================================== */

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ENABLE(ZYDIS) */
