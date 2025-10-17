/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#include "libdwarfdefs.h"

#define true                    1
#define false                   0

/* to identify a cie */
#define DW_CIE_ID 		~(0x0)
#define DW_CIE_VERSION		1 /* DWARF2 */
#define DW_CIE_VERSION3		3 /* DWARF3 */
#define DW_CIE_VERSION4		4 /* DWARF4 */
#define ABBREV_HASH_TABLE_SIZE	10


/* 
    These are allocation type codes for structs that
    are internal to the Libdwarf Consumer library.
*/
#define DW_DLA_ABBREV_LIST	DW_DLA_ADDR + 1
#define DW_DLA_CHAIN		DW_DLA_ADDR + 2
#define DW_DLA_CU_CONTEXT	DW_DLA_ADDR + 3
#define DW_DLA_FRAME		DW_DLA_ADDR + 4
#define DW_DLA_GLOBAL_CONTEXT	DW_DLA_ADDR + 5
#define DW_DLA_FILE_ENTRY	DW_DLA_ADDR + 6
#define DW_DLA_LINE_CONTEXT	DW_DLA_ADDR + 7
#define DW_DLA_LOC_CHAIN	DW_DLA_ADDR + 8
#define DW_DLA_HASH_TABLE	DW_DLA_ADDR + 9
#define DW_DLA_FUNC_CONTEXT	DW_DLA_ADDR + 10
#define DW_DLA_TYPENAME_CONTEXT	DW_DLA_ADDR + 11
#define DW_DLA_VAR_CONTEXT	DW_DLA_ADDR + 12
#define DW_DLA_WEAK_CONTEXT	DW_DLA_ADDR + 13
#define DW_DLA_PUBTYPES_CONTEXT	DW_DLA_ADDR + 14 /* DWARF3 */

/* Maximum number of allocation types for allocation routines. */
#define MAX_DW_DLA		DW_DLA_PUBTYPES_CONTEXT

/*Dwarf_Word  is unsigned word usable for index, count in memory */
/*Dwarf_Sword is   signed word usable for index, count in memory */
/* The are 32 or 64 bits depending if 64 bit longs or not, which
** fits the  ILP32 and LP64 models
** These work equally well with ILP64.
*/

typedef unsigned long Dwarf_Word;
typedef signed long Dwarf_Sword;

typedef signed char Dwarf_Sbyte;
typedef unsigned char Dwarf_Ubyte;
typedef signed short Dwarf_Shalf;
typedef Dwarf_Small *Dwarf_Byte_Ptr;

/* these 2 are fixed sizes which must not vary with the
** ILP32/LP64 model. Between these two, stay at 32 bit.
*/
typedef __uint32_t Dwarf_ufixed;
typedef __int32_t Dwarf_sfixed;

/*
        In various places the code mistakenly associates
        forms 8 bytes long with Dwarf_Signed or Dwarf_Unsigned
	This is not a very portable assumption.
        The following should be used instead for 64 bit integers.
*/
typedef __uint32_t Dwarf_ufixed64;
typedef __int32_t Dwarf_sfixed64;


typedef struct Dwarf_Abbrev_List_s *Dwarf_Abbrev_List;
typedef struct Dwarf_File_Entry_s *Dwarf_File_Entry;
typedef struct Dwarf_CU_Context_s *Dwarf_CU_Context;
typedef struct Dwarf_Hash_Table_s *Dwarf_Hash_Table;


typedef struct Dwarf_Alloc_Hdr_s *Dwarf_Alloc_Hdr;
