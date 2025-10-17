/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
/* This source file used for SGI-IRIX rqs processing.
   Unused otherwise.
*/


#include "config.h"
#include "dwarf_incl.h"
#include <stdio.h>
#include "dwarf_line.h"
#ifdef HAVE_ALLOCA_H
#include <alloca.h>
#endif



/*
	Return DW_DLV_OK or, if error,
	DW_DLV_ERROR.

	Thru pointers, return 2 arrays and a count
	for rqs.
*/
int
_dwarf_line_address_offsets(Dwarf_Debug dbg,
			    Dwarf_Die die,
			    Dwarf_Addr ** addrs,
			    Dwarf_Off ** offs,
			    Dwarf_Unsigned * returncount,
			    Dwarf_Error * err)
{
    Dwarf_Addr *laddrs;
    Dwarf_Off *loffsets;
    Dwarf_Signed lcount;
    Dwarf_Signed i;
    int res;
    Dwarf_Line *linebuf;

    res = _dwarf_internal_srclines(die, &linebuf, &lcount,	/* addrlist= 
								 */ true,
				   /* linelist= */ false, err);
    if (res != DW_DLV_OK) {
	return res;
    }
    laddrs = (Dwarf_Addr *)
	_dwarf_get_alloc(dbg, DW_DLA_ADDR, lcount);
    if (laddrs == NULL) {
	dwarf_srclines_dealloc(dbg, linebuf, lcount);
	_dwarf_error(dbg, err, DW_DLE_ALLOC_FAIL);
	return (DW_DLV_ERROR);
    }
    loffsets = (Dwarf_Off *)
	_dwarf_get_alloc(dbg, DW_DLA_ADDR, lcount);
    if (loffsets == NULL) {
	dwarf_srclines_dealloc(dbg, linebuf, lcount);
	/* We already allocated what laddrs points at, so we'e better
	   deallocate that space since we are not going to return the
	   pointer to the caller. */
	dwarf_dealloc(dbg, laddrs, DW_DLA_ADDR);
	_dwarf_error(dbg, err, DW_DLE_ALLOC_FAIL);
	return (DW_DLV_ERROR);
    }

    for (i = 0; i < lcount; i++) {
	laddrs[i] = linebuf[i]->li_address;
	loffsets[i] = linebuf[i]->li_addr_line.li_offset;
    }
    dwarf_srclines_dealloc(dbg, linebuf, lcount);
    *returncount = lcount;
    *offs = loffsets;
    *addrs = laddrs;
    return DW_DLV_OK;
}

/*
   It's impossible for callers of dwarf_srclines() to get to and
   free all the resources (in particular, the li_context and its
   lc_file_entries). 
   So this function, new July 2005, does it.  
*/
