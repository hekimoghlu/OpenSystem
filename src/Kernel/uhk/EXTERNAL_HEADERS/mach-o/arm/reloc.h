/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
 * Relocation types used in the arm implementation.  Relocation entries for
 * things other than instructions use the same generic relocation as discribed
 * in <mach-o/reloc.h> and their r_type is ARM_RELOC_VANILLA, one of the
 * *_SECTDIFF or the *_PB_LA_PTR types.  The rest of the relocation types are
 * for instructions.  Since they are for instructions the r_address field
 * indicates the 32 bit instruction that the relocation is to be preformed on.
 */
enum reloc_type_arm
{
    ARM_RELOC_VANILLA,	/* generic relocation as discribed above */
    ARM_RELOC_PAIR,	/* the second relocation entry of a pair */
    ARM_RELOC_SECTDIFF,	/* a PAIR follows with subtract symbol value */
    ARM_RELOC_LOCAL_SECTDIFF, /* like ARM_RELOC_SECTDIFF, but the symbol
				 referenced was local.  */
    ARM_RELOC_PB_LA_PTR,/* prebound lazy pointer */
    ARM_RELOC_BR24,	/* 24 bit branch displacement (to a word address) */
    ARM_THUMB_RELOC_BR22, /* 22 bit branch displacement (to a half-word
			     address) */
    ARM_THUMB_32BIT_BRANCH, /* obsolete - a thumb 32-bit branch instruction
			     possibly needing page-spanning branch workaround */

    /*
     * For these two r_type relocations they always have a pair following them
     * and the r_length bits are used differently.  The encoding of the
     * r_length is as follows:
     * low bit of r_length:
     *  0 - :lower16: for movw instructions
     *  1 - :upper16: for movt instructions
     * high bit of r_length:
     *  0 - arm instructions
     *  1 - thumb instructions   
     * the other half of the relocated expression is in the following pair
     * relocation entry in the the low 16 bits of r_address field.
     */
    ARM_RELOC_HALF,
    ARM_RELOC_HALF_SECTDIFF
};
