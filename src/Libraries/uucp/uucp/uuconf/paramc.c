/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
const char _uuconf_paramc_rcsid[] = "$Id: paramc.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

/* Handle protocol-parameter commands by inserting them into an array
   of structures.  The return value may include UUCONF_CMDTABRET_KEEP
   and UUCONF_CMDTABRET_EXIT, if appropriate.  */

int
_uuconf_iadd_proto_param (qglobal, argc, argv, pqparam, pblock)
     struct sglobal *qglobal;
     int argc;
     char **argv;
     struct uuconf_proto_param **pqparam;
     pointer pblock;
{
  struct uuconf_proto_param *q;
  size_t c;
  struct uuconf_proto_param_entry *qentry;

  if (argc < 2)
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  /* The first argument is the protocol character.  */
  if (argv[0][1] != '\0')
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  if (*pqparam == NULL)
    {
      *pqparam = ((struct uuconf_proto_param *)
		  uuconf_malloc (pblock,
				 2 * sizeof (struct uuconf_proto_param)));
      if (*pqparam == NULL)
	{
	  qglobal->ierrno = errno;
	  return (UUCONF_MALLOC_FAILED
		  | UUCONF_ERROR_ERRNO
		  | UUCONF_CMDTABRET_EXIT);
	}
      (*pqparam)[1].uuconf_bproto = '\0';
      q = *pqparam;
      q->uuconf_bproto = argv[0][0];
      q->uuconf_qentries = NULL;
    }
  else
    {
      c = 0;
      for (q = *pqparam; q->uuconf_bproto != '\0'; q++)
	{
	  if (q->uuconf_bproto == argv[0][0])
	    break;
	  ++c;
	}

      if (q->uuconf_bproto == '\0')
	{
	  struct uuconf_proto_param *qnew;

	  qnew = ((struct uuconf_proto_param *)
		  uuconf_malloc (pblock,
				 ((c + 2)
				  * sizeof (struct uuconf_proto_param))));
	  if (qnew == NULL)
	    {
	      qglobal->ierrno = errno;
	      return (UUCONF_MALLOC_FAILED
		      | UUCONF_ERROR_ERRNO
		      | UUCONF_CMDTABRET_EXIT);
	    }

	  memcpy ((pointer) qnew, (pointer) *pqparam,
		  c * sizeof (struct uuconf_proto_param));
	  qnew[c + 1].uuconf_bproto = '\0';

	  uuconf_free (pblock, *pqparam);
	  *pqparam = qnew;

	  q = qnew + c;
	  q->uuconf_bproto = argv[0][0];
	  q->uuconf_qentries = NULL;
	}
    }

  if (q->uuconf_qentries == NULL)
    {
      qentry = ((struct uuconf_proto_param_entry *)
		uuconf_malloc (pblock,
			       2 * sizeof (struct uuconf_proto_param_entry)));
      if (qentry == NULL)
	{
	  qglobal->ierrno = errno;
	  return (UUCONF_MALLOC_FAILED
		  | UUCONF_ERROR_ERRNO
		  | UUCONF_CMDTABRET_EXIT);
	}

      qentry[1].uuconf_cargs = 0;
      q->uuconf_qentries = qentry;
    }
  else
    {
      struct uuconf_proto_param_entry *qnewent;

      c = 0;
      for (qentry = q->uuconf_qentries; qentry->uuconf_cargs != 0; qentry++)
	++c;

      qnewent = ((struct uuconf_proto_param_entry *)
		 uuconf_malloc (pblock,
				((c + 2) *
				 sizeof (struct uuconf_proto_param_entry))));
      if (qnewent == NULL)
	{
	  qglobal->ierrno = errno;
	  return (UUCONF_MALLOC_FAILED
		  | UUCONF_ERROR_ERRNO
		  | UUCONF_CMDTABRET_EXIT);
	}

      memcpy ((pointer) qnewent, (pointer) q->uuconf_qentries,
	      c * sizeof (struct uuconf_proto_param_entry));
      qnewent[c + 1].uuconf_cargs = 0;

      uuconf_free (pblock, q->uuconf_qentries);
      q->uuconf_qentries = qnewent;

      qentry = qnewent + c;
    }

  qentry->uuconf_cargs = argc - 1;
  qentry->uuconf_pzargs = (char **) uuconf_malloc (pblock,
						   ((argc - 1)
						    * sizeof (char *)));
  if (qentry->uuconf_pzargs == NULL)
    {
      qglobal->ierrno = errno;
      qentry->uuconf_cargs = 0;
      return (UUCONF_MALLOC_FAILED
	      | UUCONF_ERROR_ERRNO
	      | UUCONF_CMDTABRET_EXIT);
    }
  memcpy ((pointer) qentry->uuconf_pzargs, (pointer) (argv + 1),
	  (argc - 1) * sizeof (char *));

  return UUCONF_CMDTABRET_KEEP;
}
