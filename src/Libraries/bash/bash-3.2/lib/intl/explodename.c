/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "loadinfo.h"

/* On some strange systems still no definition of NULL is found.  Sigh!  */
#ifndef NULL
# if defined __STDC__ && __STDC__
#  define NULL ((void *) 0)
# else
#  define NULL 0
# endif
#endif

/* @@ end of prolog @@ */

char *
_nl_find_language (name)
     const char *name;
{
  while (name[0] != '\0' && name[0] != '_' && name[0] != '@'
	 && name[0] != '+' && name[0] != ',')
    ++name;

  return (char *) name;
}


int
_nl_explode_name (name, language, modifier, territory, codeset,
		  normalized_codeset, special, sponsor, revision)
     char *name;
     const char **language;
     const char **modifier;
     const char **territory;
     const char **codeset;
     const char **normalized_codeset;
     const char **special;
     const char **sponsor;
     const char **revision;
{
  enum { undecided, xpg, cen } syntax;
  char *cp;
  int mask;

  *modifier = NULL;
  *territory = NULL;
  *codeset = NULL;
  *normalized_codeset = NULL;
  *special = NULL;
  *sponsor = NULL;
  *revision = NULL;

  /* Now we determine the single parts of the locale name.  First
     look for the language.  Termination symbols are `_' and `@' if
     we use XPG4 style, and `_', `+', and `,' if we use CEN syntax.  */
  mask = 0;
  syntax = undecided;
  *language = cp = name;
  cp = _nl_find_language (*language);

  if (*language == cp)
    /* This does not make sense: language has to be specified.  Use
       this entry as it is without exploding.  Perhaps it is an alias.  */
    cp = strchr (*language, '\0');
  else if (cp[0] == '_')
    {
      /* Next is the territory.  */
      cp[0] = '\0';
      *territory = ++cp;

      while (cp[0] != '\0' && cp[0] != '.' && cp[0] != '@'
	     && cp[0] != '+' && cp[0] != ',' && cp[0] != '_')
	++cp;

      mask |= TERRITORY;

      if (cp[0] == '.')
	{
	  /* Next is the codeset.  */
	  syntax = xpg;
	  cp[0] = '\0';
	  *codeset = ++cp;

	  while (cp[0] != '\0' && cp[0] != '@')
	    ++cp;

	  mask |= XPG_CODESET;

	  if (*codeset != cp && (*codeset)[0] != '\0')
	    {
	      *normalized_codeset = _nl_normalize_codeset (*codeset,
							   cp - *codeset);
	      if (strcmp (*codeset, *normalized_codeset) == 0)
		free ((char *) *normalized_codeset);
	      else
		mask |= XPG_NORM_CODESET;
	    }
	}
    }

  if (cp[0] == '@' || (syntax != xpg && cp[0] == '+'))
    {
      /* Next is the modifier.  */
      syntax = cp[0] == '@' ? xpg : cen;
      cp[0] = '\0';
      *modifier = ++cp;

      while (syntax == cen && cp[0] != '\0' && cp[0] != '+'
	     && cp[0] != ',' && cp[0] != '_')
	++cp;

      mask |= XPG_MODIFIER | CEN_AUDIENCE;
    }

  if (syntax != xpg && (cp[0] == '+' || cp[0] == ',' || cp[0] == '_'))
    {
      syntax = cen;

      if (cp[0] == '+')
	{
 	  /* Next is special application (CEN syntax).  */
	  cp[0] = '\0';
	  *special = ++cp;

	  while (cp[0] != '\0' && cp[0] != ',' && cp[0] != '_')
	    ++cp;

	  mask |= CEN_SPECIAL;
	}

      if (cp[0] == ',')
	{
 	  /* Next is sponsor (CEN syntax).  */
	  cp[0] = '\0';
	  *sponsor = ++cp;

	  while (cp[0] != '\0' && cp[0] != '_')
	    ++cp;

	  mask |= CEN_SPONSOR;
	}

      if (cp[0] == '_')
	{
 	  /* Next is revision (CEN syntax).  */
	  cp[0] = '\0';
	  *revision = ++cp;

	  mask |= CEN_REVISION;
	}
    }

  /* For CEN syntax values it might be important to have the
     separator character in the file name, not for XPG syntax.  */
  if (syntax == xpg)
    {
      if (*territory != NULL && (*territory)[0] == '\0')
	mask &= ~TERRITORY;

      if (*codeset != NULL && (*codeset)[0] == '\0')
	mask &= ~XPG_CODESET;

      if (*modifier != NULL && (*modifier)[0] == '\0')
	mask &= ~XPG_MODIFIER;
    }

  return mask;
}

