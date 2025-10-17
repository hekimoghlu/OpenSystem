/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
const char _uuconf_vport_rcsid[] = "$Id: vport.c,v 1.12 2002/03/05 19:10:43 ian Rel $";
#endif

#include <errno.h>
#include <ctype.h>

/* Find a port in the V2 configuration files by name, baud rate, and
   special purpose function.  */

int
uuconf_v2_find_port (pglobal, zname, ibaud, ihighbaud, pifn, pinfo, qport)
     pointer pglobal;
     const char *zname;
     long ibaud;
     long ihighbaud ATTRIBUTE_UNUSED;
     int (*pifn) P((struct uuconf_port *, pointer));
     pointer pinfo;
     struct uuconf_port *qport;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  FILE *e;
  char *zline;
  size_t cline;
  char **pzsplit;
  size_t csplit;
  int iret;
  int cchars;

  e = fopen (qglobal->qprocess->zv2devices, "r");
  if (e == NULL)
    {
      if (FNO_SUCH_FILE ())
	return UUCONF_NOT_FOUND;
      qglobal->ierrno = errno;
      qglobal->zfilename = qglobal->qprocess->zv2devices;
      return (UUCONF_FOPEN_FAILED
	      | UUCONF_ERROR_ERRNO
	      | UUCONF_ERROR_FILENAME);
    }

  zline = NULL;
  cline = 0;
  pzsplit = NULL;
  csplit = 0;

  iret = UUCONF_NOT_FOUND;

  qglobal->ilineno = 0;

  while ((cchars = getline (&zline, &cline, e)) > 0)
    {
      int ctoks;
      char *zend;
      long ilow, ihigh;
      pointer pblock;

      ++qglobal->ilineno;

      iret = UUCONF_NOT_FOUND;

      --cchars;
      if (zline[cchars] == '\n')
	zline[cchars] = '\0';
      zline[strcspn (zline, "#")] = '\0';

      ctoks = _uuconf_istrsplit (zline, '\0', &pzsplit, &csplit);
      if (ctoks < 0)
	{
	  qglobal->ierrno = errno;
	  iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	  break;
	}

      /* An entry in L-devices is
	 
	 type device dial-device baud dialer
	 
	 The type (normally "ACU") is treated as the name.  */

      /* If there aren't enough entries, ignore the line; this
	 should probably do something more useful.  */
      if (ctoks < 4)
	continue;

      /* Make sure the name matches any argument.  */
      if (zname != NULL
	  && strcmp (pzsplit[0], zname) != 0)
	continue;

      /* Get the baud rate.  */
      ilow = strtol (pzsplit[3], &zend, 10);
      if (*zend == '-')
	ihigh = strtol (zend + 1, (char **) NULL, 10);
      else
	ihigh = ilow;

      /* Make sure the baud rate matches any argument.  */
      if (ibaud != 0
	  && ilow != 0
	  && (ilow > ibaud || ihigh < ibaud))
	continue;

      /* Now we must construct the port information, so that we can
	 pass it to pifn.  The port type is determined by it's name,
	 unfortunately.  The name "DIR" is used for a direct port, and
	 anything else for a modem port.  */
      pblock = NULL;
      _uuconf_uclear_port (qport);
      qport->uuconf_zname = pzsplit[0];
      if (strcmp (pzsplit[0], "DIR") == 0)
	{
	  qport->uuconf_ttype = UUCONF_PORTTYPE_DIRECT;
	  qport->uuconf_u.uuconf_sdirect.uuconf_zdevice = pzsplit[1];
	  qport->uuconf_u.uuconf_sdirect.uuconf_ibaud = ilow;
	  qport->uuconf_u.uuconf_sdirect.uuconf_fcarrier = FALSE;
	  qport->uuconf_u.uuconf_sdirect.uuconf_fhardflow = TRUE;
	}
      else
	{
	  qport->uuconf_ttype = UUCONF_PORTTYPE_MODEM;
	  qport->uuconf_u.uuconf_smodem.uuconf_zdevice = pzsplit[1];
	  if (strcmp (pzsplit[2], "-") != 0)
	    qport->uuconf_u.uuconf_smodem.uuconf_zdial_device = pzsplit[2];
	  else
	    qport->uuconf_u.uuconf_smodem.uuconf_zdial_device = NULL;
	  if (ilow == ihigh)
	    {
	      qport->uuconf_u.uuconf_smodem.uuconf_ibaud = ilow;
	      qport->uuconf_u.uuconf_smodem.uuconf_ilowbaud = 0L;
	      qport->uuconf_u.uuconf_smodem.uuconf_ihighbaud = 0L;
	    }
	  else
	    {
	      qport->uuconf_u.uuconf_smodem.uuconf_ibaud = 0L;
	      qport->uuconf_u.uuconf_smodem.uuconf_ilowbaud = ilow;
	      qport->uuconf_u.uuconf_smodem.uuconf_ihighbaud = ihigh;
	    }
	  qport->uuconf_u.uuconf_smodem.uuconf_fcarrier = TRUE;
	  qport->uuconf_u.uuconf_smodem.uuconf_fhardflow = TRUE;
	  if (ctoks < 5)
	    qport->uuconf_u.uuconf_smodem.uuconf_pzdialer = NULL;
	  else
	    {
	      size_t c;
	      char **pzd;

	      /* We support dialer/token pairs, although normal V2
		 doesn't.  */
	      pblock = uuconf_malloc_block ();
	      if (pblock == NULL)
		{
		  qglobal->ierrno = errno;
		  iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
		  break;
		}
	      c = (ctoks - 4) * sizeof (char *);
	      pzd = (char **) uuconf_malloc (pblock, c + sizeof (char *));
	      if (pzd == NULL)
		{
		  qglobal->ierrno = errno;
		  uuconf_free_block (pblock);
		  iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
		  break;
		}
	      memcpy ((pointer) pzd, (pointer) (pzsplit + 4), c);
	      pzd[ctoks - 4] = NULL;

	      qport->uuconf_u.uuconf_smodem.uuconf_pzdialer = pzd;
	    }
	  qport->uuconf_u.uuconf_smodem.uuconf_qdialer = NULL;
	}

      if (pifn != NULL)
	{
	  iret = (*pifn) (qport, pinfo);
	  if (iret != UUCONF_SUCCESS)
	    {
	      if (pblock != NULL)
		uuconf_free_block (pblock);
	      if (iret != UUCONF_NOT_FOUND)
		break;
	      continue;
	    }
	}

      /* This is the port we want.  */
      if (pblock == NULL)
	{
	  pblock = uuconf_malloc_block ();
	  if (pblock == NULL)
	    {
	      qglobal->ierrno = errno;
	      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	      break;
	    }
	}

      if (uuconf_add_block (pblock, zline) != 0)
	{
	  qglobal->ierrno = errno;
	  uuconf_free_block (pblock);
	  iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	  break;
	}
      zline = NULL;

      qport->uuconf_palloc = pblock;

      break;
    }

  (void) fclose (e);

  if (zline != NULL)
    free ((pointer) zline);
  if (pzsplit != NULL)
    free ((pointer) pzsplit);

  if (iret != UUCONF_SUCCESS && iret != UUCONF_NOT_FOUND)
    {
      qglobal->zfilename = qglobal->qprocess->zv2devices;
      iret |= UUCONF_ERROR_FILENAME | UUCONF_ERROR_LINENO;
    }

  return iret;
}
