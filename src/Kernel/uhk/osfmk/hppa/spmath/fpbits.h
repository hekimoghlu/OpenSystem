/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
/*
  (c) Copyright 1986 HEWLETT-PACKARD COMPANY
  To anyone who acknowledges that this file is provided "AS IS"
  without any express or implied warranty:
      permission to use, copy, modify, and distribute this file
  for any purpose is hereby granted without fee, provided that
  the above copyright notice and this notice appears in all
  copies, and that the name of Hewlett-Packard Company not be
  used in advertising or publicity pertaining to distribution
  of the software without specific, written prior permission.
  Hewlett-Packard Company makes no representations about the
  suitability of this software for any purpose.
*/
/* @(#)fpbits.h: Revision: 1.6.88.1 Date: 93/12/07 15:06:21 */

/*
 *  These macros are designed to be portable to all machines that have
 *  a wordsize greater than or equal to 32 bits that support the portable
 *  C compiler and the standard C preprocessor.  Wordsize (default 32)
 *  and bitfield assignment (default left-to-right,  unlike VAX, PDP-11)
 *  should be predefined using the constants HOSTWDSZ and BITFRL and
 *  the C compiler "-D" flag (e.g., -DHOSTWDSZ=36 -DBITFLR for the DEC-20).
 *  Note that the macro arguments assume that the integer being referenced
 *  is a 32-bit integer (right-justified on the 20) and that bit 0 is the
 *  most significant bit.
 */

#ifndef HOSTWDSZ
#define	HOSTWDSZ	32
#endif


/*###########################  Macros  ######################################*/

/*-------------------------------------------------------------------------
 * NewDeclareBitField_Reference - Declare a structure similar to the simulator
 * function "DeclBitfR" except its use is restricted to occur within a larger
 * enclosing structure or union definition.  This declaration is an unnamed
 * structure with the argument, name, as the member name and the argument,
 * uname, as the element name.
 *----------------------------------------------------------------------- */
#define Bitfield_extract(start, length, object)		\
    ((object) >> (HOSTWDSZ - (start) - (length)) &	\
    ((unsigned)-1 >> (HOSTWDSZ - (length))))

#define Bitfield_signed_extract(start, length, object) \
    ((int)((object) << start) >> (HOSTWDSZ - (length)))

#define Bitfield_mask(start, len, object)		\
    ((object) & (((unsigned)-1 >> (HOSTWDSZ-len)) << (HOSTWDSZ-start-len)))

#define Bitfield_deposit(value,start,len,object)  object = \
    ((object) & ~(((unsigned)-1 >> (HOSTWDSZ-(len))) << (HOSTWDSZ-(start)-(len)))) | \
    (((value) & ((unsigned)-1 >> (HOSTWDSZ-(len)))) << (HOSTWDSZ-(start)-(len)))
