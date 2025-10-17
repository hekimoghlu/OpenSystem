/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/

#include <tparm_type.h>

MODULE_ID("$Id: tparm_type.c,v 1.2 2015/04/04 15:01:13 tom Exp $")

/*
 * Lookup the type of call we should make to tparm().  This ignores the actual
 * terminfo capability (bad, because it is not extensible), but makes this
 * code portable to platforms where sizeof(int) != sizeof(char *).
 */
TParams
tparm_type(const char *name)
{
#define TD(code, longname, ti, tc) \
    	{code, {longname} }, \
	{code, {ti} }, \
	{code, {tc} }
    TParams result = Numbers;
    /* *INDENT-OFF* */
    static const struct {
	TParams code;
	const char name[12];
    } table[] = {
	TD(Num_Str,	"pkey_key",	"pfkey",	"pk"),
	TD(Num_Str,	"pkey_local",	"pfloc",	"pl"),
	TD(Num_Str,	"pkey_xmit",	"pfx",		"px"),
	TD(Num_Str,	"plab_norm",	"pln",		"pn"),
	TD(Num_Str_Str, "pkey_plab",	"pfxl",		"xl"),
    };
    /* *INDENT-ON* */

    unsigned n;
    for (n = 0; n < SIZEOF(table); n++) {
	if (!strcmp(name, table[n].name)) {
	    result = table[n].code;
	    break;
	}
    }
    return result;
}
