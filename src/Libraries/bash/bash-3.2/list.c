/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
/* Copyright (C) 1996
   Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2, or (at your option) any later
   version.

   Bash is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with Bash; see the file COPYING.  If not, write to the Free Software
   Foundation, 59 Temple Place, Suite 330, Boston, MA 02111 USA. */

#include "config.h"

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "shell.h"

/* A global variable which acts as a sentinel for an `error' list return. */
GENERIC_LIST global_error_list;

#ifdef INCLUDE_UNUSED
/* Call FUNCTION on every member of LIST, a generic list. */
void
list_walk (list, function)
     GENERIC_LIST *list;
     sh_glist_func_t *function;
{
  for ( ; list; list = list->next)
    if ((*function) (list) < 0)
      return;
}

/* Call FUNCTION on every string in WORDS. */
void
wlist_walk (words, function)
     WORD_LIST *words;
     sh_icpfunc_t *function;
{
  for ( ; words; words = words->next)
    if ((*function) (words->word->word) < 0)
      return;
}
#endif /* INCLUDE_UNUSED */

/* Reverse the chain of structures in LIST.  Output the new head
   of the chain.  You should always assign the output value of this
   function to something, or you will lose the chain. */
GENERIC_LIST *
list_reverse (list)
     GENERIC_LIST *list;
{
  register GENERIC_LIST *next, *prev;

  for (prev = (GENERIC_LIST *)NULL; list; )
    {
      next = list->next;
      list->next = prev;
      prev = list;
      list = next;
    }
  return (prev);
}

/* Return the number of elements in LIST, a generic list. */
int
list_length (list)
     GENERIC_LIST *list;
{
  register int i;

  for (i = 0; list; list = list->next, i++);
  return (i);
}

/* Append TAIL to HEAD.  Return the header of the list. */
GENERIC_LIST *
list_append (head, tail)
     GENERIC_LIST *head, *tail;
{
  register GENERIC_LIST *t_head;

  if (head == 0)
    return (tail);

  for (t_head = head; t_head->next; t_head = t_head->next)
    ;
  t_head->next = tail;
  return (head);
}

#ifdef INCLUDE_UNUSED
/* Delete the element of LIST which satisfies the predicate function COMPARER.
   Returns the element that was deleted, so you can dispose of it, or -1 if
   the element wasn't found.  COMPARER is called with the list element and
   then ARG.  Note that LIST contains the address of a variable which points
   to the list.  You might call this function like this:

   SHELL_VAR *elt = list_remove (&variable_list, check_var_has_name, "foo");
   dispose_variable (elt);
*/
GENERIC_LIST *
list_remove (list, comparer, arg)
     GENERIC_LIST **list;
     Function *comparer;
     char *arg;
{
  register GENERIC_LIST *prev, *temp;

  for (prev = (GENERIC_LIST *)NULL, temp = *list; temp; prev = temp, temp = temp->next)
    {
      if ((*comparer) (temp, arg))
	{
	  if (prev)
	    prev->next = temp->next;
	  else
	    *list = temp->next;
	  return (temp);
	}
    }
  return ((GENERIC_LIST *)&global_error_list);
}
#endif
