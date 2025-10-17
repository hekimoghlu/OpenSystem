/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#include "dwarf_incl.h"
#include <stdio.h>


/*
    decode ULEB
*/
Dwarf_Unsigned
_dwarf_decode_u_leb128(Dwarf_Small * leb128, Dwarf_Word * leb128_length)
{
    unsigned char byte;
    Dwarf_Word word_number;
    Dwarf_Unsigned number;
    Dwarf_Sword shift;
    Dwarf_Sword byte_length;

    /* The following unrolls-the-loop for the first few bytes and
       unpacks into 32 bits to make this as fast as possible.
       word_number is assumed big enough that the shift has a defined
       result. */
    if ((*leb128 & 0x80) == 0) {
	if (leb128_length != NULL)
	    *leb128_length = 1;
	return (*leb128);
    } else if ((*(leb128 + 1) & 0x80) == 0) {
	if (leb128_length != NULL)
	    *leb128_length = 2;

	word_number = *leb128 & 0x7f;
	word_number |= (*(leb128 + 1) & 0x7f) << 7;
	return (word_number);
    } else if ((*(leb128 + 2) & 0x80) == 0) {
	if (leb128_length != NULL)
	    *leb128_length = 3;

	word_number = *leb128 & 0x7f;
	word_number |= (*(leb128 + 1) & 0x7f) << 7;
	word_number |= (*(leb128 + 2) & 0x7f) << 14;
	return (word_number);
    } else if ((*(leb128 + 3) & 0x80) == 0) {
	if (leb128_length != NULL)
	    *leb128_length = 4;

	word_number = *leb128 & 0x7f;
	word_number |= (*(leb128 + 1) & 0x7f) << 7;
	word_number |= (*(leb128 + 2) & 0x7f) << 14;
	word_number |= (*(leb128 + 3) & 0x7f) << 21;
	return (word_number);
    }

    /* The rest handles long numbers Because the 'number' may be larger 
       than the default int/unsigned, we must cast the 'byte' before
       the shift for the shift to have a defined result. */
    number = 0;
    shift = 0;
    byte_length = 1;
    byte = *(leb128);
    for (;;) {
	number |= ((Dwarf_Unsigned) (byte & 0x7f)) << shift;

	if ((byte & 0x80) == 0) {
	    if (leb128_length != NULL)
		*leb128_length = byte_length;
	    return (number);
	}
	shift += 7;

	byte_length++;
	++leb128;
	byte = *leb128;
    }
}

#define BITSINBYTE 8

/*
    decode SLEB
*/
Dwarf_Signed
_dwarf_decode_s_leb128(Dwarf_Small * leb128, Dwarf_Word * leb128_length)
{
    Dwarf_Signed number = 0;
    Dwarf_Bool sign = 0;
    Dwarf_Sword shift = 0;
    unsigned char byte = *leb128;
    Dwarf_Sword byte_length = 1;

    /* byte_length being the number of bytes of data absorbed so far in 
       turning the leb into a Dwarf_Signed. */

    for (;;) {
	sign = byte & 0x40;
	number |= ((Dwarf_Signed) ((byte & 0x7f))) << shift;
	shift += 7;

	if ((byte & 0x80) == 0) {
	    break;
	}
	++leb128;
	byte = *leb128;
	byte_length++;
    }

    if ((shift < sizeof(Dwarf_Signed) * BITSINBYTE) && sign) {
	number |= -((Dwarf_Signed) 1 << shift);
    }

    if (leb128_length != NULL)
	*leb128_length = byte_length;
    return (number);
}
