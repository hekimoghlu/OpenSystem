/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

int
dwarf_get_str(Dwarf_Debug dbg,
	      Dwarf_Off offset,
	      char **string,
	      Dwarf_Signed * returned_str_len, Dwarf_Error * error)
{
    int res;

    if (dbg == NULL) {
	_dwarf_error(NULL, error, DW_DLE_DBG_NULL);
	return (DW_DLV_ERROR);
    }

    if (offset == dbg->de_debug_str_size) {
	/* Normal (if we've iterated thru the set of strings using
	   dwarf_get_str and are at the end). */
	return DW_DLV_NO_ENTRY;
    }
    if (offset > dbg->de_debug_str_size) {
	_dwarf_error(dbg, error, DW_DLE_DEBUG_STR_OFFSET_BAD);
	return (DW_DLV_ERROR);
    }

    if (string == NULL) {
	_dwarf_error(dbg, error, DW_DLE_STRING_PTR_NULL);
	return (DW_DLV_ERROR);
    }

    res =
	_dwarf_load_section(dbg,
			    dbg->de_debug_str_index,
			    &dbg->de_debug_str, error);
    if (res != DW_DLV_OK) {
	return res;
    }

    *string = (char *) dbg->de_debug_str + offset;

    *returned_str_len = (strlen(*string));
    return DW_DLV_OK;
}
