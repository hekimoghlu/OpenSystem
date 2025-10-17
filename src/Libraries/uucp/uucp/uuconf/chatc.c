/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
const char _uuconf_chatc_rcsid[] = "$Id: chatc.c,v 1.10 2002/03/05 19:10:42 ian Rel $";
#endif

#include <ctype.h>
#include <errno.h>

static int icchat P((pointer pglobal, int argc, char **argv,
		     pointer pvar, pointer pinfo));
static int icchat_fail P((pointer pglobal, int argc, char **argv,
			  pointer pvar, pointer pinfo));
static int icunknown P((pointer pglobal, int argc, char **argv,
			pointer pvar, pointer pinfo));

/* The chat script commands.  */

static const struct cmdtab_offset asChat_cmds[] =
{
  { "chat", UUCONF_CMDTABTYPE_FN,
      offsetof (struct uuconf_chat, uuconf_pzchat), icchat },
  { "chat-program", UUCONF_CMDTABTYPE_FULLSTRING,
      offsetof (struct uuconf_chat, uuconf_pzprogram), NULL },
  { "chat-timeout", UUCONF_CMDTABTYPE_INT,
      offsetof (struct uuconf_chat, uuconf_ctimeout), NULL },
  { "chat-fail", UUCONF_CMDTABTYPE_FN | 2,
      offsetof (struct uuconf_chat, uuconf_pzfail), icchat_fail },
  { "chat-seven-bit", UUCONF_CMDTABTYPE_BOOLEAN,
      offsetof (struct uuconf_chat, uuconf_fstrip), NULL },
  { NULL, 0, 0, NULL }
};

#define CCHAT_CMDS (sizeof asChat_cmds / sizeof asChat_cmds[0])

/* Handle a chat script command.  The chat script commands are entered
   as UUCONF_CMDTABTYPE_PREFIX, and the commands are routed to this
   function.  We copy the command table onto the stack and repoint it
   at qchat in order to make the function reentrant.  The return value
   can include UUCONF_CMDTABRET_KEEP, but should not include
   UUCONF_CMDTABRET_EXIT.  */

int
_uuconf_ichat_cmd (qglobal, argc, argv, qchat, pblock)
     struct sglobal *qglobal;
     int argc;
     char **argv;
     struct uuconf_chat *qchat;
     pointer pblock;
{
  char *zchat;
  struct uuconf_cmdtab as[CCHAT_CMDS];
  int iret;

  /* This is only invoked when argv[0] will contain the string "chat";
     the specific chat script command comes after that point.  */
  for (zchat = argv[0]; *zchat != '\0'; zchat++)
    if ((*zchat == 'c' || *zchat == 'C')
	&& strncasecmp (zchat, "chat", sizeof "chat" - 1) == 0)
      break;
  if (*zchat == '\0')
    return UUCONF_SYNTAX_ERROR;
  argv[0] = zchat;

  _uuconf_ucmdtab_base (asChat_cmds, CCHAT_CMDS, (char *) qchat, as);

  iret = uuconf_cmd_args ((pointer) qglobal, argc, argv, as, pblock,
			  icunknown, 0, pblock);

  /* If chat-program was specified with no arguments, treat that as no
     chat-program.  This may be used to override an earlier
     chat-program.  There is a space leak here.  */
  if (qchat->uuconf_pzprogram != NULL && qchat->uuconf_pzprogram[0] == NULL)
    qchat->uuconf_pzprogram = NULL;

  return iret &~ UUCONF_CMDTABRET_EXIT;
}

/* Handle the "chat" command.  This breaks up substrings in expect
   strings, and sticks the arguments into a NULL terminated array.  */

static int
icchat (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc;
     char **argv;
     pointer pvar;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char ***ppz = (char ***) pvar;
  pointer pblock = pinfo;
  int i;

  *ppz = NULL;

  for (i = 1; i < argc; i += 2)
    {
      char *z, *zdash;
      int iret;

      /* Break the expect string into substrings.  */
      z = argv[i];
      zdash = strchr (z, '-');
      while (zdash != NULL)
	{
	  *zdash = '\0';
	  iret = _uuconf_iadd_string (qglobal, z, TRUE, FALSE, ppz,
				      pblock);
	  if (iret != UUCONF_SUCCESS)
	    return iret;
	  *zdash = '-';
	  z = zdash;
	  zdash = strchr (z + 1, '-');
	}

      iret = _uuconf_iadd_string (qglobal, z, FALSE, FALSE, ppz, pblock);
      if (iret != UUCONF_SUCCESS)
	return iret;

      /* Add the send string without breaking it up.  If it starts
	 with a dash we must replace it with an escape sequence, to
	 prevent it from being interpreted as a subsend.  */

      if (i + 1 < argc)
	{
	  if (argv[i + 1][0] != '-')
	    iret = _uuconf_iadd_string (qglobal, argv[i + 1], FALSE,
					FALSE, ppz, pblock);
	  else
	    {
	      size_t clen;

	      clen = strlen (argv[i + 1]);
	      z = uuconf_malloc (pblock, clen + 2);
	      if (z == NULL)
		{
		  qglobal->ierrno = errno;
		  return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
		}
	      z[0] = '\\';
	      memcpy ((pointer) (z + 1), (pointer) argv[i + 1], clen + 1);
	      iret = _uuconf_iadd_string (qglobal, z, FALSE, FALSE, ppz,
					  pblock);
	    }
	  if (iret != UUCONF_SUCCESS)
	    return iret;
	}
    }

  return UUCONF_CMDTABRET_KEEP;
}

/* Add a new chat failure string.  */

/*ARGSUSED*/
static int
icchat_fail (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char ***ppz = (char ***) pvar;
  pointer pblock = pinfo;

  return _uuconf_iadd_string (qglobal, argv[1], TRUE, FALSE, ppz, pblock);
}

/* Return a syntax error for an unknown command.  */

/*ARGSUSED*/
static int
icunknown (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal ATTRIBUTE_UNUSED;
     int argc ATTRIBUTE_UNUSED;
     char **argv ATTRIBUTE_UNUSED;
     pointer pvar ATTRIBUTE_UNUSED;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  return UUCONF_SYNTAX_ERROR;
}
