/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
 * alloc_entry.c -- allocation functions for terminfo entries
 *
 *	_nc_copy_entry()
 *	_nc_init_entry()
 *	_nc_merge_entry()
 *	_nc_save_str()
 *	_nc_wrap_entry()
 *
 */

#include <curses.priv.h>

#include <tic.h>

MODULE_ID("$Id: alloc_entry.c,v 1.58 2013/08/17 19:20:38 tom Exp $")

#define ABSENT_OFFSET    -1
#define CANCELLED_OFFSET -2

#ifndef __APPLE__
/* Removed as part of remedy for CVE-2022-29458 */
#define MAX_STRTAB	4096	/* documented maximum entry size */
#endif

static char *stringbuf;		/* buffer for string capabilities */
static size_t next_free;	/* next free character in stringbuf */

NCURSES_EXPORT(void)
_nc_init_entry(TERMTYPE *const tp)
/* initialize a terminal type data block */
{
#if NO_LEAKS
    if (tp == 0) {
	if (stringbuf != 0) {
	    FreeAndNull(stringbuf);
	}
	return;
    }
#endif

    if (stringbuf == 0)
#ifdef __APPLE__
	TYPE_CALLOC(char, (size_t) MAX_ENTRY_SIZE, stringbuf);
#else
	TYPE_MALLOC(char, (size_t) MAX_STRTAB, stringbuf);
#endif

    next_free = 0;

    _nc_init_termtype(tp);
}

NCURSES_EXPORT(ENTRY *)
_nc_copy_entry(ENTRY * oldp)
{
    ENTRY *newp = typeCalloc(ENTRY, 1);

    if (newp != 0) {
	*newp = *oldp;
	_nc_copy_termtype(&(newp->tterm), &(oldp->tterm));
    }
    return newp;
}

/* save a copy of string in the string buffer */
NCURSES_EXPORT(char *)
_nc_save_str(const char *const string)
{
    char *result = 0;
    size_t old_next_free = next_free;
    size_t len = strlen(string) + 1;

    if (len == 1 && next_free != 0) {
	/*
	 * Cheat a little by making an empty string point to the end of the
	 * previous string.
	 */
#ifdef __APPLE__
	if (next_free < MAX_ENTRY_SIZE) {
#else
	if (next_free < MAX_STRTAB) {
#endif
	    result = (stringbuf + next_free - 1);
	}
#ifdef __APPLE__
    } else if (next_free + len < MAX_ENTRY_SIZE) {
	_nc_STRCPY(&stringbuf[next_free], string, MAX_ENTRY_SIZE);
#else
    } else if (next_free + len < MAX_STRTAB) {
	_nc_STRCPY(&stringbuf[next_free], string, MAX_STRTAB);
#endif
	DEBUG(7, ("Saved string %s", _nc_visbuf(string)));
	DEBUG(7, ("at location %d", (int) next_free));
	next_free += len;
	result = (stringbuf + old_next_free);
    } else {
	_nc_warning("Too much data, some is lost: %s", string);
    }
    return result;
}

NCURSES_EXPORT(void)
_nc_wrap_entry(ENTRY * const ep, bool copy_strings)
/* copy the string parts to allocated storage, preserving pointers to it */
{
    int offsets[MAX_ENTRY_SIZE / sizeof(short)];
    int useoffsets[MAX_USES];
    unsigned i, n;
    unsigned nuses = ep->nuses;
    TERMTYPE *tp = &(ep->tterm);

    if (copy_strings) {
	next_free = 0;		/* clear static storage */

	/* copy term_names, Strings, uses */
	tp->term_names = _nc_save_str(tp->term_names);
	for_each_string(i, tp) {
	    if (tp->Strings[i] != ABSENT_STRING &&
		tp->Strings[i] != CANCELLED_STRING) {
		tp->Strings[i] = _nc_save_str(tp->Strings[i]);
	    }
	}

	for (i = 0; i < nuses; i++) {
	    if (ep->uses[i].name == 0) {
		ep->uses[i].name = _nc_save_str(ep->uses[i].name);
	    }
	}

	free(tp->str_table);
    }

    assert(tp->term_names >= stringbuf);
    n = (unsigned) (tp->term_names - stringbuf);
    for_each_string(i, &(ep->tterm)) {
	if (i < SIZEOF(offsets)) {
	    if (tp->Strings[i] == ABSENT_STRING) {
		offsets[i] = ABSENT_OFFSET;
	    } else if (tp->Strings[i] == CANCELLED_STRING) {
		offsets[i] = CANCELLED_OFFSET;
	    } else {
		offsets[i] = (int) (tp->Strings[i] - stringbuf);
	    }
	}
    }

    for (i = 0; i < nuses; i++) {
	if (ep->uses[i].name == 0)
	    useoffsets[i] = ABSENT_OFFSET;
	else
	    useoffsets[i] = (int) (ep->uses[i].name - stringbuf);
    }

    TYPE_MALLOC(char, next_free, tp->str_table);
    (void) memcpy(tp->str_table, stringbuf, next_free);

    tp->term_names = tp->str_table + n;
    for_each_string(i, &(ep->tterm)) {
	if (i < SIZEOF(offsets)) {
	    if (offsets[i] == ABSENT_OFFSET) {
		tp->Strings[i] = ABSENT_STRING;
	    } else if (offsets[i] == CANCELLED_OFFSET) {
		tp->Strings[i] = CANCELLED_STRING;
	    } else {
		tp->Strings[i] = tp->str_table + offsets[i];
	    }
	}
    }

#if NCURSES_XNAMES
    if (!copy_strings) {
	if ((n = (unsigned) NUM_EXT_NAMES(tp)) != 0) {
	    if (n < SIZEOF(offsets)) {
		size_t length = 0;
		size_t offset;
		for (i = 0; i < n; i++) {
		    length += strlen(tp->ext_Names[i]) + 1;
		    offsets[i] = (int) (tp->ext_Names[i] - stringbuf);
		}
		TYPE_MALLOC(char, length, tp->ext_str_table);
		for (i = 0, offset = 0; i < n; i++) {
		    tp->ext_Names[i] = tp->ext_str_table + offset;
		    _nc_STRCPY(tp->ext_Names[i],
			       stringbuf + offsets[i],
			       length - offset);
		    offset += strlen(tp->ext_Names[i]) + 1;
		}
	    }
	}
    }
#endif

    for (i = 0; i < nuses; i++) {
	if (useoffsets[i] == ABSENT_OFFSET)
	    ep->uses[i].name = 0;
	else
	    ep->uses[i].name = (tp->str_table + useoffsets[i]);
    }
}

NCURSES_EXPORT(void)
_nc_merge_entry(TERMTYPE *const to, TERMTYPE *const from)
/* merge capabilities from `from' entry into `to' entry */
{
    unsigned i;

#if NCURSES_XNAMES
    _nc_align_termtype(to, from);
#endif
    for_each_boolean(i, from) {
	if (to->Booleans[i] != (char) CANCELLED_BOOLEAN) {
	    int mergebool = from->Booleans[i];

	    if (mergebool == CANCELLED_BOOLEAN)
		to->Booleans[i] = FALSE;
	    else if (mergebool == TRUE)
		to->Booleans[i] = (char) mergebool;
	}
    }

    for_each_number(i, from) {
	if (to->Numbers[i] != CANCELLED_NUMERIC) {
	    short mergenum = from->Numbers[i];

	    if (mergenum == CANCELLED_NUMERIC)
		to->Numbers[i] = ABSENT_NUMERIC;
	    else if (mergenum != ABSENT_NUMERIC)
		to->Numbers[i] = mergenum;
	}
    }

    /*
     * Note: the copies of strings this makes don't have their own
     * storage.  This is OK right now, but will be a problem if we
     * we ever want to deallocate entries.
     */
    for_each_string(i, from) {
	if (to->Strings[i] != CANCELLED_STRING) {
	    char *mergestring = from->Strings[i];

	    if (mergestring == CANCELLED_STRING)
		to->Strings[i] = ABSENT_STRING;
	    else if (mergestring != ABSENT_STRING)
		to->Strings[i] = mergestring;
	}
    }
}

#if NO_LEAKS
NCURSES_EXPORT(void)
_nc_alloc_entry_leaks(void)
{
    if (stringbuf != 0) {
	FreeAndNull(stringbuf);
    }
    next_free = 0;
}
#endif
