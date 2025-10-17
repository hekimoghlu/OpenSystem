/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#include "ZydisInternalSharedData.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

/* ============================================================================================== */
/* Data tables                                                                                    */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Instruction definitions                                                                        */
/* ---------------------------------------------------------------------------------------------- */

#ifdef ZYDIS_MINIMAL_MODE
#   define ZYDIS_NOTMIN(x)
#else
#   define ZYDIS_NOTMIN(x) , x
#endif

#include "ZydisGeneratedInstructionDefinitions.inc"

#undef ZYDIS_NOTMIN

/* ---------------------------------------------------------------------------------------------- */
/* Operand definitions                                                                            */
/* ---------------------------------------------------------------------------------------------- */

#define ZYDIS_OPERAND_DEFINITION(type, encoding, access) \
    { type, encoding, access }

#include "ZydisGeneratedOperandDefinitions.inc"

#undef ZYDIS_OPERAND_DEFINITION

/* ---------------------------------------------------------------------------------------------- */
/* Accessed CPU flags                                                                             */
/* ---------------------------------------------------------------------------------------------- */

#include "ZydisGeneratedAccessedFlags.inc"

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */
/* Functions                                                                                      */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Instruction definition                                                                         */
/* ---------------------------------------------------------------------------------------------- */

void ZydisGetInstructionDefinition(ZydisInstructionEncoding encoding, ZyanU16 id,
    const ZydisInstructionDefinition** definition)
{
    switch (encoding)
    {
    case ZYDIS_INSTRUCTION_ENCODING_LEGACY:
        *definition = (const ZydisInstructionDefinition*)&ISTR_DEFINITIONS_LEGACY[id];
        break;
    case ZYDIS_INSTRUCTION_ENCODING_3DNOW:
        *definition = (const ZydisInstructionDefinition*)&ISTR_DEFINITIONS_3DNOW[id];
        break;
    case ZYDIS_INSTRUCTION_ENCODING_XOP:
        *definition = (const ZydisInstructionDefinition*)&ISTR_DEFINITIONS_XOP[id];
        break;
    case ZYDIS_INSTRUCTION_ENCODING_VEX:
        *definition = (const ZydisInstructionDefinition*)&ISTR_DEFINITIONS_VEX[id];
        break;
#ifndef ZYDIS_DISABLE_AVX512
    case ZYDIS_INSTRUCTION_ENCODING_EVEX:
        *definition = (const ZydisInstructionDefinition*)&ISTR_DEFINITIONS_EVEX[id];
        break;
#endif
#ifndef ZYDIS_DISABLE_KNC
    case ZYDIS_INSTRUCTION_ENCODING_MVEX:
        *definition = (const ZydisInstructionDefinition*)&ISTR_DEFINITIONS_MVEX[id];
        break;
#endif
    default:
        ZYAN_UNREACHABLE;
    }
}

/* ---------------------------------------------------------------------------------------------- */
/* Operand definition                                                                             */
/* ---------------------------------------------------------------------------------------------- */

#ifndef ZYDIS_MINIMAL_MODE

#if defined(ZYAN_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif

ZyanU8 ZydisGetOperandDefinitions(const ZydisInstructionDefinition* definition,
    const ZydisOperandDefinition** operand)
{
    if (definition->operand_count == 0)
    {
        *operand = ZYAN_NULL;
        return 0;
    }
    ZYAN_ASSERT(definition->operand_reference != 0xFFFF);
    *operand = &OPERAND_DEFINITIONS[definition->operand_reference];
    return definition->operand_count;
}

#if defined(ZYAN_GCC)
#pragma GCC diagnostic pop
#endif

#endif

/* ---------------------------------------------------------------------------------------------- */
/* Element info                                                                                   */
/* ---------------------------------------------------------------------------------------------- */

#ifndef ZYDIS_MINIMAL_MODE
void ZydisGetElementInfo(ZydisInternalElementType element, ZydisElementType* type,
    ZydisElementSize* size)
{
    static const struct
    {
        ZydisElementType type;
        ZydisElementSize size;
    } lookup[ZYDIS_IELEMENT_TYPE_MAX_VALUE + 1] =
    {
        { ZYDIS_ELEMENT_TYPE_INVALID  ,   0 },
        { ZYDIS_ELEMENT_TYPE_INVALID  ,   0 },
        { ZYDIS_ELEMENT_TYPE_STRUCT   ,   0 },
        { ZYDIS_ELEMENT_TYPE_INT      ,   0 },
        { ZYDIS_ELEMENT_TYPE_UINT     ,   0 },
        { ZYDIS_ELEMENT_TYPE_INT      ,   1 },
        { ZYDIS_ELEMENT_TYPE_INT      ,   8 },
        { ZYDIS_ELEMENT_TYPE_INT      ,  16 },
        { ZYDIS_ELEMENT_TYPE_INT      ,  32 },
        { ZYDIS_ELEMENT_TYPE_INT      ,  64 },
        { ZYDIS_ELEMENT_TYPE_UINT     ,   8 },
        { ZYDIS_ELEMENT_TYPE_UINT     ,  16 },
        { ZYDIS_ELEMENT_TYPE_UINT     ,  32 },
        { ZYDIS_ELEMENT_TYPE_UINT     ,  64 },
        { ZYDIS_ELEMENT_TYPE_UINT     , 128 },
        { ZYDIS_ELEMENT_TYPE_UINT     , 256 },
        { ZYDIS_ELEMENT_TYPE_FLOAT16  ,  16 },
        { ZYDIS_ELEMENT_TYPE_FLOAT16  ,  32 }, // TODO: Should indicate 2 float16 elements
        { ZYDIS_ELEMENT_TYPE_FLOAT32  ,  32 },
        { ZYDIS_ELEMENT_TYPE_FLOAT64  ,  64 },
        { ZYDIS_ELEMENT_TYPE_FLOAT80  ,  80 },
        { ZYDIS_ELEMENT_TYPE_LONGBCD  ,  80 },
        { ZYDIS_ELEMENT_TYPE_CC       ,   3 },
        { ZYDIS_ELEMENT_TYPE_CC       ,   5 }
    };

    ZYAN_ASSERT((ZyanUSize)element < ZYAN_ARRAY_LENGTH(lookup));

    *type = lookup[element].type;
    *size = lookup[element].size;
}
#endif

/* ---------------------------------------------------------------------------------------------- */
/* Accessed CPU flags                                                                             */
/* ---------------------------------------------------------------------------------------------- */

#ifndef ZYDIS_MINIMAL_MODE
ZyanBool ZydisGetAccessedFlags(const ZydisInstructionDefinition* definition,
    const ZydisAccessedFlags** flags)
{
    ZYAN_ASSERT(definition->flags_reference < ZYAN_ARRAY_LENGTH(ACCESSED_FLAGS));
    *flags = &ACCESSED_FLAGS[definition->flags_reference];
    return (definition->flags_reference != 0);
}
#endif

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ENABLE(ZYDIS) */
