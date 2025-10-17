/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#include "dwarf_weaks.h"
#include "dwarf_global.h"

int
dwarf_get_weaks(Dwarf_Debug dbg,
		Dwarf_Weak ** weaks,
		Dwarf_Signed * ret_weak_count, Dwarf_Error * error)
{
    int res;

    res =
	_dwarf_load_section(dbg,
			    dbg->de_debug_weaknames_index,
			    &dbg->de_debug_weaknames, error);
    if (res != DW_DLV_OK) {
	return res;
    }

    return _dwarf_internal_get_pubnames_like_data(dbg, dbg->de_debug_weaknames, dbg->de_debug_weaknames_size, (Dwarf_Global **) weaks,	/* type 
																	   punning, 
																	   Dwarf_Type 
																	   is 
																	   never 
																	   a 
																	   completed 
																	   type 
																	 */
						  ret_weak_count,
						  error,
						  DW_DLA_WEAK_CONTEXT,
						  DW_DLA_WEAK,
						  DW_DLE_DEBUG_WEAKNAMES_LENGTH_BAD,
						  DW_DLE_DEBUG_WEAKNAMES_VERSION_ERROR);

}

/* Deallocating fully requires deallocating the list
   and all entries.  But some internal data is
   not exposed, so we need a function with internal knowledge.
*/

void
dwarf_weaks_dealloc(Dwarf_Debug dbg, Dwarf_Weak * dwgl,
		    Dwarf_Signed count)
{
    _dwarf_internal_globals_dealloc(dbg, (Dwarf_Global *) dwgl,
				    count,
				    DW_DLA_WEAK_CONTEXT,
				    DW_DLA_WEAK, DW_DLA_LIST);
    return;
}



int
dwarf_weakname(Dwarf_Weak weak_in, char **ret_name, Dwarf_Error * error)
{
    Dwarf_Global weak = (Dwarf_Global) weak_in;

    if (weak == NULL) {
	_dwarf_error(NULL, error, DW_DLE_WEAK_NULL);
	return (DW_DLV_ERROR);
    }
    *ret_name = (char *) (weak->gl_name);
    return DW_DLV_OK;
}


int
dwarf_weak_die_offset(Dwarf_Weak weak_in,
		      Dwarf_Off * weak_off, Dwarf_Error * error)
{
    Dwarf_Global weak = (Dwarf_Global) weak_in;

    return dwarf_global_die_offset(weak, weak_off, error);
}


int
dwarf_weak_cu_offset(Dwarf_Weak weak_in,
		     Dwarf_Off * weak_off, Dwarf_Error * error)
{
    Dwarf_Global weak = (Dwarf_Global) weak_in;

    return dwarf_global_cu_offset(weak, weak_off, error);
}


int
dwarf_weak_name_offsets(Dwarf_Weak weak_in,
			char **weak_name,
			Dwarf_Off * die_offset,
			Dwarf_Off * cu_offset, Dwarf_Error * error)
{
    Dwarf_Global weak = (Dwarf_Global) weak_in;

    return dwarf_global_name_offsets(weak,
				     weak_name,
				     die_offset, cu_offset, error);
}
