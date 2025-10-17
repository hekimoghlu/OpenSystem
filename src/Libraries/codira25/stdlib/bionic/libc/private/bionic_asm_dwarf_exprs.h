/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
// Define assembler macros for generating DWARF CFI instructions that use DWARF expressions.
// Assemblers don't natively support DWARF expressions, so use the C preprocessor and assembler
// macros to lower them to .cfi_escape directives.
//
// Signal trampolines need to use DWARF expressions to record the locations of saved registers,
// because the offsets from the restored SP to the saved registers is variable. e.g. A signal frame
// can have optional FP/SIMD extensions, and there may be extra padding if the interrupted SP wasn't
// aligned.

// DWARF constants.
#define DW_CFA_def_cfa_expression 0x0f
#define DW_CFA_expression 0x10
#define DW_OP_breg0 0x70
#define DW_OP_deref 0x06

// Return the size of a small uleb128 value: either 1 or 2 bytes
#define ULEB128_14BIT_SIZE(val) \
  (1 + (((val) > 0x7f) & 1))

// Return the size of a small sleb128 value: either 1 or 2 bytes
#define SLEB128_14BIT_SIZE(val)       \
  (1 + (((val) < -0x40) & 1) +        \
       (((val) > 0x3f) & 1)     )

// Output a 1 or 2-byte CFI uleb128 absolute value.
.macro m_cfi_uleb128 val
  .if (\val) < 0 || (\val) > 0x3fff
    .error "m_cfi_uleb128 value is out of range (\val)"
  .elseif (\val) > 0x7f
    .cfi_escape ((\val) & 0x7f) | 0x80
    .cfi_escape (\val) >> 7
  .else
    .cfi_escape (\val)
  .endif
.endm

// Output a 1 or 2-byte CFI sleb128 absolute value.
.macro m_cfi_sleb128 val
  .if (\val) < -0x2000 || (\val) > 0x1fff
    .error "m_cfi_sleb128 value is out of range (\val)"
  .elseif (\val) < -0x40 || (\val) > 0x3f
    .cfi_escape ((\val) & 0x7f) | 0x80
    .cfi_escape ((\val) >> 7) & 0x7f
  .else
    .cfi_escape (\val) & 0x7f
  .endif
.endm

.macro check_base_reg reg_no
  .if (\reg_no) < 0 || (\reg_no) > 31
    .error "base register is out of range for DW_OP_breg0..DW_OP_breg31 (\reg_no)"
  .endif
.endm

// Set CFA to the expression, *(base_reg + offset)
.macro m_cfi_def_cfa_deref base_reg, offset
  check_base_reg (\base_reg)
  .cfi_escape DW_CFA_def_cfa_expression
  m_cfi_uleb128 (1 + SLEB128_14BIT_SIZE(\offset) + 1)   // size of DWARF expression in bytes
  .cfi_escape DW_OP_breg0 + (\base_reg)                 // expr: 1 byte
  m_cfi_sleb128 (\offset)                               // expr: 1 or 2 bytes
  .cfi_escape DW_OP_deref                               // expr: 1 byte
.endm

// Set the address of the register's previous value to the expression, (base_reg + offset)
.macro m_cfi_breg_offset dest_reg, base_reg, offset
  check_base_reg (\base_reg)
  .cfi_escape DW_CFA_expression
  m_cfi_uleb128 (\dest_reg)
  m_cfi_uleb128 (1 + SLEB128_14BIT_SIZE(\offset)) // size of DWARF expression in bytes
  .cfi_escape DW_OP_breg0 + (\base_reg)           // expr: 1 byte
  m_cfi_sleb128 (\offset)                         // expr: 1 or 2 bytes
.endm
