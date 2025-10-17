/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
/* return codes for functions
*/
#define DW_DLV_NO_ENTRY -1
#define DW_DLV_OK        0
#define DW_DLV_ERROR     1


/* the following are the 'section' number passed to the called-back
   function.
   The called-back application must translate this to the 
   appropriate elf section number/pointer.

   Putting this burden on the application avoids having to store
   the numbers in the Dwarf_Debug structure (thereby saving space
   for most consumers).
*/
#define DW_SECTION_INFO      0
#define DW_SECTION_FRAME     1
#define DW_SECTION_ARANGES   2
#define DW_SECTION_LINE      3
#define DW_SECTION_LOC       4  /* .debug_loc */

/* section is one of the above codes: it specifies a section.
   secoff is the offset in the dwarf section.
   existingAddr is the value at the specified offset (so the
	called back routine can sanity check the proceedings).
   It's up to the caller to know the size of an address (4 or 8)
   and update the right number of bytes.
*/
typedef int (*Dwarf_addr_callback_func)   (int /*section*/, 
        Dwarf_Off /*secoff*/, Dwarf_Addr /*existingAddr*/);

/* call this to do the work: it calls back thru cb_func
   once per each address to be modified.
   Once this returns you are done.
   Returns DW_DLV_OK if finished ok.
   Returns DW_DLV_ERROR if there was some kind of error, in which
	the dwarf error number was passed back thu the dwerr ptr.
   Returns DW_DLV_NO_ENTRY if there are no relevant dwarf sections,
	so there were no addresses to be modified (and none
	called back).
*/
int _dwarf_addr_finder(dwarf_elf_handle elf_file_ptr,
                Dwarf_addr_callback_func cb_func,
                int *dwerr);

