/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
/* Reads DWARF3 .debug_pubtypes section. */


#include "config.h"
#include "dwarf_incl.h"
#include <stdio.h>
#include "dwarf_types.h"
#include "dwarf_global.h"

int
dwarf_get_pubtypes(Dwarf_Debug dbg,
		   Dwarf_Type ** types,
		   Dwarf_Signed * ret_type_count, Dwarf_Error * error)
{
    int res;

    res =
	_dwarf_load_section(dbg,
			    dbg->de_debug_pubtypes_index,
			    &dbg->de_debug_pubtypes, error);
    if (res != DW_DLV_OK) {
	return res;
    }

    return _dwarf_internal_get_pubnames_like_data(dbg, dbg->de_debug_pubtypes, dbg->de_debug_pubtypes_size, (Dwarf_Global **) types,	/* type 
																	   punning,
																	   Dwarf_Type 
																	   is never
																	   a
																	   completed 
																	   type */
						  ret_type_count, error, DW_DLA_PUBTYPES_CONTEXT, DW_DLA_GLOBAL,	/* We 
															   don't 
															   have 
															   DW_DLA_PUBTYPES,
															   so use
															   DW_DLA_GLOBAL. */
						  DW_DLE_DEBUG_PUBTYPES_LENGTH_BAD,
						  DW_DLE_DEBUG_PUBTYPES_VERSION_ERROR);
}

/* Deallocating fully requires deallocating the list
   and all entries.  But some internal data is
   not exposed, so we need a function with internal knowledge.
*/

void
dwarf_pubtypes_dealloc(Dwarf_Debug dbg, Dwarf_Type * dwgl,
		       Dwarf_Signed count)
{
    _dwarf_internal_globals_dealloc(dbg, (Dwarf_Global *) dwgl, count, DW_DLA_PUBTYPES_CONTEXT, DW_DLA_GLOBAL,	/* We 
														   don't 
														   have 
														   DW_DLA_PUBTYPES,
														   so use
														   DW_DLA_GLOBAL. */
				    DW_DLA_LIST);
    return;
}



int
dwarf_pubtypename(Dwarf_Type type_in, char **ret_name,
		  Dwarf_Error * error)
{
    Dwarf_Global type = (Dwarf_Global) type_in;

    if (type == NULL) {
	_dwarf_error(NULL, error, DW_DLE_TYPE_NULL);
	return (DW_DLV_ERROR);
    }

    *ret_name = (char *) (type->gl_name);
    return DW_DLV_OK;
}


int
dwarf_pubtype_cu_offset(Dwarf_Type type_in,
			Dwarf_Off * ret_offset, Dwarf_Error * error)
{
    Dwarf_Global type = (Dwarf_Global) type_in;

    return dwarf_global_cu_offset(type, ret_offset, error);

}


int
dwarf_pubtype_name_offsets(Dwarf_Type type_in,
			   char **returned_name,
			   Dwarf_Off * die_offset,
			   Dwarf_Off * cu_die_offset,
			   Dwarf_Error * error)
{
    Dwarf_Global type = (Dwarf_Global) type_in;

    return dwarf_global_name_offsets(type,
				     returned_name,
				     die_offset, cu_die_offset, error);
}
