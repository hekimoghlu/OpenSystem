/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
const char _uuconf_tdialc_rcsid[] = "$Id: tdialc.c,v 1.9 2002/03/05 19:10:43 ian Rel $";
#endif

static int idchat P((pointer pglobal, int argc, char **argv, pointer pvar,
		     pointer pinfo));
static int iddtr_toggle P((pointer pglobal, int argc, char **argv,
			   pointer pvar, pointer pinfo));
static int idcomplete P((pointer pglobal, int argc, char **argv,
			 pointer pvar, pointer pinfo));
static int idproto_param P((pointer pglobal, int argc, char **argv,
			    pointer pvar, pointer pinfo));
static int idcunknown P((pointer pglobal, int argc, char **argv,
			 pointer pvar, pointer pinfo));

/* The command table for dialer commands.  The "dialer" command is
   handled specially.  */
static const struct cmdtab_offset asDialer_cmds[] =
{
  { "chat", UUCONF_CMDTABTYPE_PREFIX | 0,
      offsetof (struct uuconf_dialer, uuconf_schat), idchat },
  { "dialtone", UUCONF_CMDTABTYPE_STRING,
      offsetof (struct uuconf_dialer, uuconf_zdialtone), NULL },
  { "pause", UUCONF_CMDTABTYPE_STRING,
      offsetof (struct uuconf_dialer, uuconf_zpause), NULL },
  { "carrier", UUCONF_CMDTABTYPE_BOOLEAN,
      offsetof (struct uuconf_dialer, uuconf_fcarrier), NULL },
  { "carrier-wait", UUCONF_CMDTABTYPE_INT,
      offsetof (struct uuconf_dialer, uuconf_ccarrier_wait), NULL },
  { "dtr-toggle", UUCONF_CMDTABTYPE_FN | 0, (size_t) -1, iddtr_toggle },
  { "complete", UUCONF_CMDTABTYPE_FN | 2,
      offsetof (struct uuconf_dialer, uuconf_scomplete), idcomplete },
  { "complete-chat", UUCONF_CMDTABTYPE_PREFIX,
      offsetof (struct uuconf_dialer, uuconf_scomplete), idchat },
  { "abort", UUCONF_CMDTABTYPE_FN | 2,
      offsetof (struct uuconf_dialer, uuconf_sabort), idcomplete },
  { "abort-chat", UUCONF_CMDTABTYPE_PREFIX,
      offsetof (struct uuconf_dialer, uuconf_sabort), idchat },
  { "protocol-parameter", UUCONF_CMDTABTYPE_FN | 0,
      offsetof (struct uuconf_dialer, uuconf_qproto_params), idproto_param },
  { "seven-bit", UUCONF_CMDTABTYPE_FN | 2,
      offsetof (struct uuconf_dialer, uuconf_ireliable), _uuconf_iseven_bit },
  { "reliable", UUCONF_CMDTABTYPE_FN | 2,
      offsetof (struct uuconf_dialer, uuconf_ireliable), _uuconf_ireliable },
  { "half-duplex", UUCONF_CMDTABTYPE_FN | 2,
      offsetof (struct uuconf_dialer, uuconf_ireliable),
      _uuconf_ihalf_duplex },
  { NULL, 0, 0, NULL }
};

#define CDIALER_CMDS (sizeof asDialer_cmds / sizeof asDialer_cmds[0])

/* Handle a command passed to a dialer from a Taylor UUCP
   configuration file.  This can be called when reading the dialer
   file, the port file, or the sys file.  The return value may have
   UUCONF_CMDTABRET_KEEP set, but not UUCONF_CMDTABRET_EXIT.  It
   assigns values to the elements of qdialer.  The first time this is
   called, qdialer->uuconf_palloc should be set.  This will not set
   qdialer->uuconf_zname.  */

int
_uuconf_idialer_cmd (qglobal, argc, argv, qdialer)
     struct sglobal *qglobal;
     int argc;
     char **argv;
     struct uuconf_dialer *qdialer;
{
  struct uuconf_cmdtab as[CDIALER_CMDS];
  int iret;

  _uuconf_ucmdtab_base (asDialer_cmds, CDIALER_CMDS, (char *) qdialer, as);

  iret = uuconf_cmd_args ((pointer) qglobal, argc, argv, as,
			  (pointer) qdialer, idcunknown, 0,
			  qdialer->uuconf_palloc);

  return iret &~ UUCONF_CMDTABRET_EXIT;
}

/* Reroute a chat script command.  */

static int
idchat (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc;
     char **argv;
     pointer pvar;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct uuconf_chat *qchat = (struct uuconf_chat *) pvar;
  struct uuconf_dialer *qdialer = (struct uuconf_dialer *) pinfo;

  return _uuconf_ichat_cmd (qglobal, argc, argv, qchat,
			    qdialer->uuconf_palloc);
}

/* Handle the "dtr-toggle" command, which may take two arguments.  */

/*ARGSUSED*/
static int
iddtr_toggle (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc;
     char **argv;
     pointer pvar ATTRIBUTE_UNUSED;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct uuconf_dialer *qdialer = (struct uuconf_dialer *) pinfo;
  int iret;

  if (argc < 2 || argc > 3)
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  iret = _uuconf_iboolean (qglobal, argv[1], &qdialer->uuconf_fdtr_toggle);
  if ((iret &~ UUCONF_CMDTABRET_KEEP) != UUCONF_SUCCESS)
    return iret;

  if (argc < 3)
    return iret;

  iret |= _uuconf_iboolean (qglobal, argv[2],
			    &qdialer->uuconf_fdtr_toggle_wait);

  return iret;
}

/* Handle the "complete" and "abort" commands.  These just turn a
   string into a trivial chat script.  */

/*ARGSUSED*/
static int 
idcomplete (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct uuconf_chat *qchat = (struct uuconf_chat *) pvar;
  struct uuconf_dialer *qdialer = (struct uuconf_dialer *) pinfo;
  char *azargs[3];

  azargs[0] = (char *) "complete-chat";
  azargs[1] = (char *) "\"\"";
  azargs[2] = (char *) argv[1];

  return _uuconf_ichat_cmd (qglobal, 3, azargs, qchat,
			    qdialer->uuconf_palloc);
}

/* Handle the "protocol-parameter" command.  */

static int
idproto_param (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc;
     char **argv;
     pointer pvar;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct uuconf_proto_param **pqparam = (struct uuconf_proto_param **) pvar;
  struct uuconf_dialer *qdialer = (struct uuconf_dialer *) pinfo;

  return _uuconf_iadd_proto_param (qglobal, argc - 1, argv + 1, pqparam, 
				   qdialer->uuconf_palloc);
}

/* Give an error for an unknown dialer command.  */

/*ARGSUSED*/
static int
idcunknown (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal ATTRIBUTE_UNUSED;
     int argc ATTRIBUTE_UNUSED;
     char **argv ATTRIBUTE_UNUSED;
     pointer pvar ATTRIBUTE_UNUSED;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;
}
