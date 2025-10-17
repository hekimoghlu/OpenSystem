/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#define ASM_GLOBAL_DIRECTIVE .globl


#define C_SYMBOL_NAME(name) name
/* Macro for a label.  */
#ifdef	__STDC__
#define C_LABEL(name)		name##:
#else
#define C_LABEL(name)		name/**/:
#endif

/* This seems to always be the case on PPC.  */
#define ALIGNARG(log2) log2
/* For ELF we need the `.type' directive to make shared libs work right.  */
#define ASM_TYPE_DIRECTIVE(name,typearg) .type name,typearg;
#define ASM_SIZE_DIRECTIVE(name) .size name,.-name

/* If compiled for profiling, call `_mcount' at the start of each function.  */
#ifdef	PROF
/* The mcount code relies on the return address being on the stack
   to locate our caller and so it can restore it; so store one just
   for its benefit.  */
#ifdef PIC
#define CALL_MCOUNT							      \
  .pushsection;								      \
  .section ".data";							      \
  .align ALIGNARG(2);							      \
0:.long 0;								      \
  .previous;								      \
  mflr  %r0;								      \
  stw   %r0,4(%r1);							      \
  bl    _GLOBAL_OFFSET_TABLE_@local-4;					      \
  mflr  %r11;								      \
  lwz   %r0,0b@got(%r11);						      \
  bl    JUMPTARGET(_mcount);
#else  /* PIC */
#define CALL_MCOUNT							      \
  .section ".data";							      \
  .align ALIGNARG(2);							      \
0:.long 0;								      \
  .previous;								      \
  mflr  %r0;								      \
  lis   %r11,0b@ha;							      \
  stw   %r0,4(%r1);							      \
  addi  %r0,%r11,0b@l;							      \
  bl    JUMPTARGET(_mcount);
#endif /* PIC */
#else  /* PROF */
#define CALL_MCOUNT		/* Do nothing.  */
#endif /* PROF */

#define	ENTRY(name)							      \
  ASM_GLOBAL_DIRECTIVE C_SYMBOL_NAME(name);				      \
  ASM_TYPE_DIRECTIVE (C_SYMBOL_NAME(name),@function)			      \
  .align ALIGNARG(2);							      \
  C_LABEL(name)								      \
  CALL_MCOUNT

#define EALIGN_W_0  /* No words to insert.  */
#define EALIGN_W_1  nop
#define EALIGN_W_2  nop;nop
#define EALIGN_W_3  nop;nop;nop
#define EALIGN_W_4  EALIGN_W_3;nop
#define EALIGN_W_5  EALIGN_W_4;nop
#define EALIGN_W_6  EALIGN_W_5;nop
#define EALIGN_W_7  EALIGN_W_6;nop

/* EALIGN is like ENTRY, but does alignment to 'words'*4 bytes
   past a 2^align boundary.  */
#ifdef PROF
#define EFFI_ALIGN(name, alignt, words)					      \
  ASM_GLOBAL_DIRECTIVE C_SYMBOL_NAME(name);				      \
  ASM_TYPE_DIRECTIVE (C_SYMBOL_NAME(name),@function)			      \
  .align ALIGNARG(2);							      \
  C_LABEL(name)								      \
  CALL_MCOUNT								      \
  b 0f;									      \
  .align ALIGNARG(alignt);						      \
  EALIGN_W_##words;							      \
  0:
#else /* PROF */
#define EFFI_ALIGN(name, alignt, words)					      \
  ASM_GLOBAL_DIRECTIVE C_SYMBOL_NAME(name);				      \
  ASM_TYPE_DIRECTIVE (C_SYMBOL_NAME(name),@function)			      \
  .align ALIGNARG(alignt);						      \
  EALIGN_W_##words;							      \
  C_LABEL(name)
#endif

#define END(name)							      \
  ASM_SIZE_DIRECTIVE(name)

#ifdef PIC
#define JUMPTARGET(name) name##@plt
#else
#define JUMPTARGET(name) name
#endif

/* Local labels stripped out by the linker.  */
#define L(x) .L##x
