/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
 * Implements the `INTEL` style instruction-formatter.
 */

#ifndef ZYDIS_FORMATTER_INTEL_H
#define ZYDIS_FORMATTER_INTEL_H

#include "ZydisFormatter.h"
#include "ZydisInternalFormatterBase.h"
#include "ZydisInternalString.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ZYAN_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#elif defined(ZYAN_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

/* ============================================================================================== */
/* Formatter functions                                                                            */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Intel                                                                                          */
/* ---------------------------------------------------------------------------------------------- */

ZyanStatus ZydisFormatterIntelFormatInstruction(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

ZyanStatus ZydisFormatterIntelFormatOperandMEM(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

ZyanStatus ZydisFormatterIntelPrintMnemonic(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

ZyanStatus ZydisFormatterIntelPrintRegister(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context, ZydisRegister reg);

ZyanStatus ZydisFormatterIntelPrintDISP(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

ZyanStatus ZydisFormatterIntelPrintTypecast(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

/* ---------------------------------------------------------------------------------------------- */
/* MASM                                                                                           */
/* ---------------------------------------------------------------------------------------------- */

ZyanStatus ZydisFormatterIntelFormatInstructionMASM(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

ZyanStatus ZydisFormatterIntelPrintAddressMASM(const ZydisFormatter* formatter,
    ZydisFormatterBuffer* buffer, ZydisFormatterContext* context);

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */
/* Fomatter presets                                                                               */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* INTEL                                                                                          */
/* ---------------------------------------------------------------------------------------------- */

/**
 * The default formatter configuration for `INTEL` style disassembly.
 */
static const ZydisFormatter FORMATTER_INTEL =
{
    /* style                   */ ZYDIS_FORMATTER_STYLE_INTEL,
    /* force_memory_size       */ ZYAN_FALSE,
    /* force_memory_seg        */ ZYAN_FALSE,
    /* force_memory_scale      */ ZYAN_TRUE,
    /* force_relative_branches */ ZYAN_FALSE,
    /* force_relative_riprel   */ ZYAN_FALSE,
    /* print_branch_size       */ ZYAN_FALSE,
    /* detailed_prefixes       */ ZYAN_FALSE,
    /* addr_base               */ ZYDIS_NUMERIC_BASE_HEX,
    /* addr_signedness         */ ZYDIS_SIGNEDNESS_SIGNED,
    /* addr_padding_absolute   */ ZYDIS_PADDING_AUTO,
    /* addr_padding_relative   */ 2,
    /* disp_base               */ ZYDIS_NUMERIC_BASE_HEX,
    /* disp_signedness         */ ZYDIS_SIGNEDNESS_SIGNED,
    /* disp_padding            */ 2,
    /* imm_base                */ ZYDIS_NUMERIC_BASE_HEX,
    /* imm_signedness          */ ZYDIS_SIGNEDNESS_UNSIGNED,
    /* imm_padding             */ 2,
    /* case_prefixes           */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_mnemonic           */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_registers          */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_typecasts          */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_decorators         */ ZYDIS_LETTER_CASE_DEFAULT,
    /* hex_uppercase           */ ZYAN_TRUE,
    /* number_format           */
    {
        // ZYDIS_NUMERIC_BASE_DEC
        {
            // Prefix
            {
                /* string      */ ZYAN_NULL,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW(""),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            },
            // Suffix
            {
                /* string      */ ZYAN_NULL,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW(""),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            }
        },
        // ZYDIS_NUMERIC_BASE_HEX
        {
            // Prefix
            {
                /* string      */ &FORMATTER_INTEL.number_format[
                                      ZYDIS_NUMERIC_BASE_HEX][0].string_data,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW("0x"),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            },
            // Suffix
            {
                /* string      */ ZYAN_NULL,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW(""),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            }
        }
    },
    /* func_pre_instruction    */ ZYAN_NULL,
    /* func_post_instruction   */ ZYAN_NULL,
    /* func_format_instruction */ &ZydisFormatterIntelFormatInstruction,
    /* func_pre_operand        */ ZYAN_NULL,
    /* func_post_operand       */ ZYAN_NULL,
    /* func_format_operand_reg */ &ZydisFormatterBaseFormatOperandREG,
    /* func_format_operand_mem */ &ZydisFormatterIntelFormatOperandMEM,
    /* func_format_operand_ptr */ &ZydisFormatterBaseFormatOperandPTR,
    /* func_format_operand_imm */ &ZydisFormatterBaseFormatOperandIMM,
    /* func_print_mnemonic     */ &ZydisFormatterIntelPrintMnemonic,
    /* func_print_register     */ &ZydisFormatterIntelPrintRegister,
    /* func_print_address_abs  */ &ZydisFormatterBasePrintAddressABS,
    /* func_print_address_rel  */ &ZydisFormatterBasePrintAddressREL,
    /* func_print_disp         */ &ZydisFormatterIntelPrintDISP,
    /* func_print_imm          */ &ZydisFormatterBasePrintIMM,
    /* func_print_typecast     */ &ZydisFormatterIntelPrintTypecast,
    /* func_print_segment      */ &ZydisFormatterBasePrintSegment,
    /* func_print_prefixes     */ &ZydisFormatterBasePrintPrefixes,
    /* func_print_decorator    */ &ZydisFormatterBasePrintDecorator
};

/* ---------------------------------------------------------------------------------------------- */
/* MASM                                                                                           */
/* ---------------------------------------------------------------------------------------------- */

/**
 * The default formatter configuration for `MASM` style disassembly.
 */
static const ZydisFormatter FORMATTER_INTEL_MASM =
{
    /* style                   */ ZYDIS_FORMATTER_STYLE_INTEL_MASM,
    /* force_memory_size       */ ZYAN_TRUE,
    /* force_memory_seg        */ ZYAN_FALSE,
    /* force_memory_scale      */ ZYAN_TRUE,
    /* force_relative_branches */ ZYAN_FALSE,
    /* force_relative_riprel   */ ZYAN_FALSE,
    /* print_branch_size       */ ZYAN_FALSE,
    /* detailed_prefixes       */ ZYAN_FALSE,
    /* addr_base               */ ZYDIS_NUMERIC_BASE_HEX,
    /* addr_signedness         */ ZYDIS_SIGNEDNESS_SIGNED,
    /* addr_padding_absolute   */ ZYDIS_PADDING_DISABLED,
    /* addr_padding_relative   */ ZYDIS_PADDING_DISABLED,
    /* disp_base               */ ZYDIS_NUMERIC_BASE_HEX,
    /* disp_signedness         */ ZYDIS_SIGNEDNESS_SIGNED,
    /* disp_padding            */ ZYDIS_PADDING_DISABLED,
    /* imm_base                */ ZYDIS_NUMERIC_BASE_HEX,
    /* imm_signedness          */ ZYDIS_SIGNEDNESS_AUTO,
    /* imm_padding             */ ZYDIS_PADDING_DISABLED,
    /* case_prefixes           */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_mnemonic           */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_registers          */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_typecasts          */ ZYDIS_LETTER_CASE_DEFAULT,
    /* case_decorators         */ ZYDIS_LETTER_CASE_DEFAULT,
    /* hex_uppercase           */ ZYAN_TRUE,
    /* number_format           */
    {
        // ZYDIS_NUMERIC_BASE_DEC
        {
            // Prefix
            {
                /* string      */ ZYAN_NULL,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW(""),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            },
            // Suffix
            {
                /* string      */ ZYAN_NULL,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW(""),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            }
        },
        // ZYDIS_NUMERIC_BASE_HEX
        {
            // Prefix
            {
                /* string      */ ZYAN_NULL,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW(""),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            },
            // Suffix
            {
                /* string      */ &FORMATTER_INTEL_MASM.number_format[
                                      ZYDIS_NUMERIC_BASE_HEX][1].string_data,
                /* string_data */ ZYAN_DEFINE_STRING_VIEW("h"),
                /* buffer      */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            }
        }
    },
    /* func_pre_instruction    */ ZYAN_NULL,
    /* func_post_instruction   */ ZYAN_NULL,
    /* func_format_instruction */ &ZydisFormatterIntelFormatInstructionMASM,
    /* func_pre_operand        */ ZYAN_NULL,
    /* func_post_operand       */ ZYAN_NULL,
    /* func_format_operand_reg */ &ZydisFormatterBaseFormatOperandREG,
    /* func_format_operand_mem */ &ZydisFormatterIntelFormatOperandMEM,
    /* func_format_operand_ptr */ &ZydisFormatterBaseFormatOperandPTR,
    /* func_format_operand_imm */ &ZydisFormatterBaseFormatOperandIMM,
    /* func_print_mnemonic     */ &ZydisFormatterIntelPrintMnemonic,
    /* func_print_register     */ &ZydisFormatterIntelPrintRegister,
    /* func_print_address_abs  */ &ZydisFormatterIntelPrintAddressMASM,
    /* func_print_address_rel  */ &ZydisFormatterIntelPrintAddressMASM,
    /* func_print_disp         */ &ZydisFormatterIntelPrintDISP,
    /* func_print_imm          */ &ZydisFormatterBasePrintIMM,
    /* func_print_typecast     */ &ZydisFormatterIntelPrintTypecast,
    /* func_print_segment      */ &ZydisFormatterBasePrintSegment,
    /* func_print_prefixes     */ &ZydisFormatterBasePrintPrefixes,
    /* func_print_decorator    */ &ZydisFormatterBasePrintDecorator
};

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */

#if defined(ZYAN_CLANG)
#pragma clang diagnostic pop
#elif defined(ZYAN_GCC)
#pragma GCC diagnostic pop
#endif

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ZYDIS_FORMATTER_INTEL_H
