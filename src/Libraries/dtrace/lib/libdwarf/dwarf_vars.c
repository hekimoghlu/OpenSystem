/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#include "dwarf_vars.h"
#include "dwarf_global.h"

int
dwarf_get_vars(Dwarf_Debug dbg,
	       Dwarf_Var ** vars,
	       Dwarf_Signed * ret_var_count, Dwarf_Error * error)
{
    int res;

    res =
	_dwarf_load_section(dbg,
			    dbg->de_debug_varnames_index,
			    &dbg->de_debug_varnames, error);
    if (res != DW_DLV_OK) {
	return res;
    }

    return _dwarf_internal_get_pubnames_like_data(dbg, dbg->de_debug_varnames, dbg->de_debug_varnames_size, (Dwarf_Global **) vars,	/* type
																	   punning,
																	   Dwarf_Type
																	   is never a
																	   completed
																	   type */
						  ret_var_count,
						  error,
						  DW_DLA_VAR_CONTEXT,
						  DW_DLA_VAR,
						  DW_DLE_DEBUG_VARNAMES_LENGTH_BAD,
						  DW_DLE_DEBUG_VARNAMES_VERSION_ERROR);
}

/* Deallocating fully requires deallocating the list
   and all entries.  But some internal data is
   not exposed, so we need a function with internal knowledge.
*/

void
dwarf_vars_dealloc(Dwarf_Debug dbg, Dwarf_Var * dwgl,
		   Dwarf_Signed count)
{
    _dwarf_internal_globals_dealloc(dbg, (Dwarf_Global *) dwgl,
				    count,
				    DW_DLA_VAR_CONTEXT,
				    DW_DLA_VAR, DW_DLA_LIST);
    return;
}


int
dwarf_varname(Dwarf_Var var_in, char **ret_varname, Dwarf_Error * error)
{
    Dwarf_Global var = (Dwarf_Global) var_in;

    if (var == NULL) {
	_dwarf_error(NULL, error, DW_DLE_VAR_NULL);
	return (DW_DLV_ERROR);
    }

    *ret_varname = (char *) (var->gl_name);
    return DW_DLV_OK;
}


int
dwarf_var_die_offset(Dwarf_Var var_in,
		     Dwarf_Off * returned_offset, Dwarf_Error * error)
{
    Dwarf_Global var = (Dwarf_Global) var_in;

    return dwarf_global_die_offset(var, returned_offset, error);

}


int
dwarf_var_cu_offset(Dwarf_Var var_in,
		    Dwarf_Off * returned_offset, Dwarf_Error * error)
{
    Dwarf_Global var = (Dwarf_Global) var_in;

    return dwarf_global_cu_offset(var, returned_offset, error);
}


int
dwarf_var_name_offsets(Dwarf_Var var_in,
		       char **returned_name,
		       Dwarf_Off * die_offset,
		       Dwarf_Off * cu_offset, Dwarf_Error * error)
{
    Dwarf_Global var = (Dwarf_Global) var_in;

    return
	dwarf_global_name_offsets(var,
				  returned_name, die_offset, cu_offset,
				  error);
}
