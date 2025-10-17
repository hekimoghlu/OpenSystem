/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#include "../config.h"

/*
 * On Solaris 8 there's a clash between "bool" in curses and RPC.
 * We don't need curses here, so ensure it doesn't get included.
 */
#define ZSH_NO_TERM_HANDLING

#include "zsh.mdh"
#include "hashnameddir.pro"

/****************************************/
/* Named Directory Hash Table Functions */
/****************************************/

/* hash table containing named directories */

/**/
mod_export HashTable nameddirtab;

/* != 0 if all the usernames have already been *
 * added to the named directory hash table.    */

static int allusersadded;

/* Create new hash table for named directories */

/**/
void
createnameddirtable(void)
{
    nameddirtab = newhashtable(201, "nameddirtab", NULL);

    nameddirtab->hash        = hasher;
    nameddirtab->emptytable  = emptynameddirtable;
    nameddirtab->filltable   = fillnameddirtable;
    nameddirtab->cmpnodes    = strcmp;
    nameddirtab->addnode     = addnameddirnode;
    nameddirtab->getnode     = gethashnode;
    nameddirtab->getnode2    = gethashnode2;
    nameddirtab->removenode  = removenameddirnode;
    nameddirtab->disablenode = NULL;
    nameddirtab->enablenode  = NULL;
    nameddirtab->freenode    = freenameddirnode;
    nameddirtab->printnode   = printnameddirnode;

    allusersadded = 0;
    finddir(NULL);		/* clear the finddir cache */
}

/* Empty the named directories table */

/**/
static void
emptynameddirtable(HashTable ht)
{
    emptyhashtable(ht);
    allusersadded = 0;
    finddir(NULL);		/* clear the finddir cache */
}

/* Add all the usernames in the password file/database *
 * to the named directories table.                     */

/**/
static void
fillnameddirtable(UNUSED(HashTable ht))
{
    if (!allusersadded) {
#ifdef USE_GETPWENT
	struct passwd *pw;

	setpwent();

	/* loop through the password file/database *
	 * and add all entries returned.           */
	while ((pw = getpwent()) && !errflag)
	    adduserdir(pw->pw_name, pw->pw_dir, ND_USERNAME, 1);

	endpwent();
#endif /* USE_GETPWENT */
	allusersadded = 1;
    }
}

/* Add an entry to the named directory hash *
 * table, clearing the finddir() cache and  *
 * initialising the `diff' member.          */

/**/
static void
addnameddirnode(HashTable ht, char *nam, void *nodeptr)
{
    Nameddir nd = (Nameddir) nodeptr;

    nd->diff = strlen(nd->dir) - strlen(nam);
    finddir(NULL);		/* clear the finddir cache */
    addhashnode(ht, nam, nodeptr);
}

/* Remove an entry from the named directory  *
 * hash table, clearing the finddir() cache. */

/**/
static HashNode
removenameddirnode(HashTable ht, const char *nam)
{
    HashNode hn = removehashnode(ht, nam);

    if(hn)
	finddir(NULL);		/* clear the finddir cache */
    return hn;
}

/* Free up the memory used by a named directory hash node. */

/**/
static void
freenameddirnode(HashNode hn)
{
    Nameddir nd = (Nameddir) hn;

    zsfree(nd->node.nam);
    zsfree(nd->dir);
    zfree(nd, sizeof(struct nameddir));
}

/* Print a named directory */

/**/
static void
printnameddirnode(HashNode hn, int printflags)
{
    Nameddir nd = (Nameddir) hn;

    if (printflags & PRINT_NAMEONLY) {
	zputs(nd->node.nam, stdout);
	putchar('\n');
	return;
    }

    if (printflags & PRINT_LIST) {
      printf("hash -d ");

      if(nd->node.nam[0] == '-')
	    printf("-- ");
    }

    quotedzputs(nd->node.nam, stdout);
    putchar('=');
    quotedzputs(nd->dir, stdout);
    putchar('\n');
}

#include "../config.h"
