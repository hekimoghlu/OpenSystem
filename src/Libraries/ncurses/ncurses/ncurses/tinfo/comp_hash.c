/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
/****************************************************************************
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

/*
 *	comp_hash.c --- Routines to deal with the hashtable of capability
 *			names.
 *
 */

#define USE_TERMLIB 1
#include <curses.priv.h>

#include <tic.h>
#include <hashsize.h>

MODULE_ID("$Id: comp_hash.c,v 1.48 2009/08/08 17:36:21 tom Exp $")

/*
 * Finds the entry for the given string in the hash table if present.
 * Returns a pointer to the entry in the table or 0 if not found.
 */
/* entrypoint used by tack (do not alter) */
NCURSES_EXPORT(struct name_table_entry const *)
_nc_find_entry(const char *string,
	       const HashValue * hash_table)
{
    bool termcap = (hash_table != _nc_get_hash_table(FALSE));
    const HashData *data = _nc_get_hash_info(termcap);
    int hashvalue;
    struct name_table_entry const *ptr = 0;
    struct name_table_entry const *real_table;

    hashvalue = data->hash_of(string);

    if (data->table_data[hashvalue] >= 0) {

	real_table = _nc_get_table(termcap);
	ptr = real_table + data->table_data[hashvalue];
	while (!data->compare_names(ptr->nte_name, string)) {
	    if (ptr->nte_link < 0) {
		ptr = 0;
		break;
	    }
	    ptr = real_table + (ptr->nte_link
				+ data->table_data[data->table_size]);
	}
    }

    return (ptr);
}

/*
 * Finds the entry for the given name with the given type in the given table if
 * present (as distinct from _nc_find_entry, which finds the last entry
 * regardless of type).
 *
 * Returns a pointer to the entry in the table or 0 if not found.
 */
NCURSES_EXPORT(struct name_table_entry const *)
_nc_find_type_entry(const char *string,
		    int type,
		    bool termcap)
{
    struct name_table_entry const *ptr = NULL;
    const HashData *data = _nc_get_hash_info(termcap);
    int hashvalue = data->hash_of(string);

    if (data->table_data[hashvalue] >= 0) {
	const struct name_table_entry *const table = _nc_get_table(termcap);

	ptr = table + data->table_data[hashvalue];
	while (ptr->nte_type != type
	       || !data->compare_names(ptr->nte_name, string)) {
	    if (ptr->nte_link < 0) {
		ptr = 0;
		break;
	    }
	    ptr = table + (ptr->nte_link + data->table_data[data->table_size]);
	}
    }

    return ptr;
}
