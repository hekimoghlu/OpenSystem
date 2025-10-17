/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#include "uucnfi.h"

#if USE_RCS_ID
const char _uuconf_cmdarg_rcsid[] = "$Id: cmdarg.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <ctype.h>

#undef strcmp
#if HAVE_STRCASECMP
#undef strcasecmp
#endif
extern int strcmp (), strcasecmp ();

/* Look up a command with arguments in a table and execute it.  */

int
uuconf_cmd_args (pglobal, cargs, pzargs, qtab, pinfo, pfiunknown, iflags,
		 pblock)
     pointer pglobal;
     int cargs;
     char **pzargs;
     const struct uuconf_cmdtab *qtab;
     pointer pinfo;
     int (*pfiunknown) P((pointer, int, char **, pointer, pointer));
     int iflags;
     pointer pblock;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int bfirstu, bfirstl;
  int (*pficmp) P((const char *, const char *));
  register const struct uuconf_cmdtab *q;
  int itype;
  int callowed;

  bfirstu = bfirstl = pzargs[0][0];
  if ((iflags & UUCONF_CMDTABFLAG_CASE) != 0)
    pficmp = strcmp;
  else
    {
      if (islower (bfirstu))
	bfirstu = toupper (bfirstu);
      if (isupper (bfirstl))
	bfirstl = tolower (bfirstl);
      pficmp = strcasecmp;
    }

  itype = 0;

  for (q = qtab; q->uuconf_zcmd != NULL; q++)
    {
      int bfirst;

      bfirst = q->uuconf_zcmd[0];
      if (bfirst != bfirstu && bfirst != bfirstl)
	continue;

      itype = UUCONF_TTYPE_CMDTABTYPE (q->uuconf_itype);
      if (itype != UUCONF_CMDTABTYPE_PREFIX)
	{
	  if ((*pficmp) (q->uuconf_zcmd, pzargs[0]) == 0)
	    break;
	}
      else
	{
	  size_t clen;

	  clen = strlen (q->uuconf_zcmd);
	  if ((iflags & UUCONF_CMDTABFLAG_CASE) != 0)
	    {
	      if (strncmp (q->uuconf_zcmd, pzargs[0], clen) == 0)
		break;
	    }
	  else
	    {
	      if (strncasecmp (q->uuconf_zcmd, pzargs[0], clen) == 0)
		break;
	    }
	}
    }

  if (q->uuconf_zcmd == NULL)
    {
      if (pfiunknown == NULL)
	return UUCONF_CMDTABRET_CONTINUE;
      return (*pfiunknown) (pglobal, cargs, pzargs, (pointer) NULL, pinfo);
    }

  callowed = UUCONF_CARGS_CMDTABTYPE (q->uuconf_itype);
  if (callowed != 0 && callowed != cargs)
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  switch (itype)
    {
    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_STRING):
      if (cargs == 1)
	*(char **) q->uuconf_pvar = (char *) "";
      else if (cargs == 2)
	*(char **) q->uuconf_pvar = pzargs[1];
      else
	return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

      return UUCONF_CMDTABRET_KEEP;

    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_INT):
      return _uuconf_iint (qglobal, pzargs[1], q->uuconf_pvar, TRUE);

    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_LONG):
      return _uuconf_iint (qglobal, pzargs[1], q->uuconf_pvar, FALSE);

    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_BOOLEAN):
      return _uuconf_iboolean (qglobal, pzargs[1], (int *) q->uuconf_pvar);

    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_FULLSTRING):
      if (cargs == 1)
	{
	  char ***ppz = (char ***) q->uuconf_pvar;
	  int iret;
	  
	  *ppz = NULL;
	  iret = _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				      ppz, pblock);
	  if (iret != UUCONF_SUCCESS)
	    return iret | UUCONF_CMDTABRET_EXIT;

	  return UUCONF_CMDTABRET_CONTINUE;
	}
      else
	{
	  char ***ppz = (char ***) q->uuconf_pvar;
	  int i;

	  *ppz = NULL;
	  for (i = 1; i < cargs; i++)
	    {
	      int iret;

	      iret = _uuconf_iadd_string (qglobal, pzargs[i], FALSE, FALSE,
					  ppz, pblock);
	      if (iret != UUCONF_SUCCESS)
		{
		  *ppz = NULL;
		  return iret | UUCONF_CMDTABRET_EXIT;
		}
	    }

	  return UUCONF_CMDTABRET_KEEP;
	}

    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_FN):
    case UUCONF_TTYPE_CMDTABTYPE (UUCONF_CMDTABTYPE_PREFIX):
      return (*q->uuconf_pifn) (pglobal, cargs, pzargs, q->uuconf_pvar,
				pinfo);

    default:
      return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;
    }

  /*NOTREACHED*/
}
