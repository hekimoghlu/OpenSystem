/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

static const ZydisInstructionEncodingInfo INSTR_ENCODINGS[] =
{
    { 0, { { 0, 0, 0 } }, { { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM, { { 0, 0, 0 } }, { { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_DISP, { { 16, 32, 64 } }, { { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 8, 8, 8 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 16, 16, 16 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 16, 32, 64 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 8, 8, 8 }, ZYAN_TRUE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 16, 32, 32 }, ZYAN_TRUE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 8, 8, 8 }, ZYAN_TRUE, ZYAN_TRUE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 16, 32, 32 }, ZYAN_TRUE, ZYAN_TRUE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 32, 32, 32 }, ZYAN_TRUE, ZYAN_TRUE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 8, 8, 8 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 32, 32, 32 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 8, 8, 8 }, ZYAN_TRUE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 16, 32, 32 }, ZYAN_TRUE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_HAS_IMM0, { { 0, 0, 0 } }, { { { 16, 32, 32 }, ZYAN_TRUE, ZYAN_TRUE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_FORCE_REG_FORM, { { 0, 0, 0 } }, { { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE }, { { 0, 0, 0 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0 | ZYDIS_INSTR_ENC_FLAG_HAS_IMM1, { { 0, 0, 0 } }, { { { 16, 16, 16 }, ZYAN_FALSE, ZYAN_FALSE }, { { 8, 8, 8 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_IMM0 | ZYDIS_INSTR_ENC_FLAG_HAS_IMM1, { { 0, 0, 0 } }, { { { 16, 32, 32 }, ZYAN_FALSE, ZYAN_FALSE }, { { 16, 16, 16 }, ZYAN_FALSE, ZYAN_FALSE } } }, 
    { ZYDIS_INSTR_ENC_FLAG_HAS_MODRM | ZYDIS_INSTR_ENC_FLAG_HAS_IMM0 | ZYDIS_INSTR_ENC_FLAG_HAS_IMM1, { { 0, 0, 0 } }, { { { 8, 8, 8 }, ZYAN_FALSE, ZYAN_FALSE }, { { 8, 8, 8 }, ZYAN_FALSE, ZYAN_FALSE } } }
};
