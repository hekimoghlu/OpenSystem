/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
 * @brief
 */

#ifndef ZYDIS_METAINFO_H
#define ZYDIS_METAINFO_H

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

#include "ZydisGeneratedEnumInstructionCategory.h"
#include "ZydisGeneratedEnumISASet.h"
#include "ZydisGeneratedEnumISAExt.h"

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

 /**
 * Returns the specified instruction category string.
 *
 * @param   category    The instruction category.
 *
 * @return  The instruction category string or `ZYAN_NULL`, if an invalid category was passed.
 */
ZYDIS_EXPORT const char* ZydisCategoryGetString(ZydisInstructionCategory category);

/**
 * Returns the specified isa-set string.
 *
 * @param   isa_set The isa-set.
 *
 * @return  The isa-set string or `ZYAN_NULL`, if an invalid isa-set was passed.
 */
ZYDIS_EXPORT const char* ZydisISASetGetString(ZydisISASet isa_set);

/**
 * Returns the specified isa-extension string.
 *
 * @param   isa_ext The isa-extension.
 *
 * @return  The isa-extension string or `ZYAN_NULL`, if an invalid isa-extension was passed.
 */
ZYDIS_EXPORT const char* ZydisISAExtGetString(ZydisISAExt isa_ext);

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYDIS_METAINFO_H */
