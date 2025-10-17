/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
const char _uuconf_hport_rcsid[] = "$Id: hport.c,v 1.18 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>
#include <ctype.h>

/* Find a port in the HDB configuration files by name, baud rate, and
   special purpose function.  */

int
uuconf_hdb_find_port (pglobal, zname, ibaud, ihighbaud, pifn, pinfo, qport)
     pointer pglobal;
     const char *zname;
     long ibaud;
     long ihighbaud ATTRIBUTE_UNUSED;
     int (*pifn) P((struct uuconf_port *, pointer));
     pointer pinfo;
     struct uuconf_port *qport;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char *zline;
  size_t cline;
  char **pzsplit;
  size_t csplit;
  int iret;
  char **pz;

  zline = NULL;
  cline = 0;
  pzsplit = NULL;
  csplit = 0;

  iret = UUCONF_NOT_FOUND;

  for (pz = qglobal->qprocess->pzhdb_devices; *pz != NULL; pz++)
    {
      FILE *e;
      int cchars;

      qglobal->ilineno = 0;

      e = fopen (*pz, "r");
      if (e == NULL)
	{
	  if (FNO_SUCH_FILE ())
	    continue;
	  qglobal->ierrno = errno;
	  iret = UUCONF_FOPEN_FAILED | UUCONF_ERROR_ERRNO;
	  break;
	}

      iret = UUCONF_NOT_FOUND;

      while ((cchars = _uuconf_getline (qglobal, &zline, &cline, e)) > 0)
	{
	  int ctoks;
	  char *z, *zprotos, *zport;
	  long ilow, ihigh;
	  pointer pblock;
	  char ***ppzdialer;

	  ++qglobal->ilineno;

	  iret = UUCONF_NOT_FOUND;

	  --cchars;
	  if (zline[cchars] == '\n')
	    zline[cchars] = '\0';
	  if (isspace (BUCHAR (zline[0])) || zline[0] == '#')
	    continue;

	  ctoks = _uuconf_istrsplit (zline, '\0', &pzsplit, &csplit);
	  if (ctoks < 0)
	    {
	      qglobal->ierrno = errno;
	      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	      break;
	    }

	  /* An entry in Devices is

	     type device dial-device baud dialer-token pairs

	     The type (normally "ACU") is treated as the name.  */

	  /* If there aren't enough entries, ignore the line; this
	     should probably do something more useful.  */
	  if (ctoks < 4)
	    continue;

	  /* There may be a comma separated list of protocols after
	     the name.  */
	  zprotos = strchr (pzsplit[0], ',');
	  if (zprotos != NULL)
	    {
	      *zprotos = '\0';
	      ++zprotos;
	    }

	  zport = pzsplit[0];

	  /* Get any modem class, and pick up the baud rate while
	     we're at it.  The modem class will be appended to the
	     name, so we need to get it before we see if we've found
	     the port with the right name.  */
	  z = pzsplit[3];
	  if (strcasecmp (z, "Any") == 0
	      || strcmp (z, "-") == 0)
	    {
	      ilow = 0L;
	      ihigh = 0L;
	    }
	  else
	    {
	      char *zend;

	      while (*z != '\0' && ! isdigit (BUCHAR (*z)))
		++z;

	      ilow = strtol (z, &zend, 10);
	      if (*zend == '-')
		ihigh = strtol (zend + 1, (char **) NULL, 10);
	      else
		ihigh = ilow;

	      if (z != pzsplit[3])
		{
		  size_t cclass, cport;

		  cclass = z - pzsplit[3];
		  cport = strlen (pzsplit[0]);
		  zport = malloc (cport + cclass + 1);
		  if (zport == NULL)
		    {
		      qglobal->ierrno = errno;
		      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
		      break;
		    }
		  memcpy ((pointer) zport, (pointer) pzsplit[0], cport);
		  memcpy ((pointer) (zport + cport), (pointer) pzsplit[3],
			  cclass);
		  zport[cport + cclass] = '\0';
		}
	    }

	  /* Make sure the name and baud rate match any argument.  */
	  if ((zname != NULL
	       && strcmp (zport, zname) != 0)
	      || (ibaud != 0
		  && ilow != 0
		  && (ilow > ibaud || ihigh < ibaud)))
	    {
	      if (zport != pzsplit[0])
		free ((pointer) zport);
	      continue;
	    }

	  /* Some systems permit ,M after the device name.  This means
	     to open the port with O_NDELAY and then change it.  We
	     just ignore this flag, although perhaps we should record
	     it somewhere.  */
	  pzsplit[1][strcspn (pzsplit[1], ",")] = '\0';

	  /* Now we must construct the port information, so that we
	     can pass it to pifn.  The port type is determined by its
	     name, unfortunately.  The name "Direct" is used for a
	     direct port, "TCP" for a TCP port, and anything else for
	     a modem port.  */
	  pblock = NULL;
	  _uuconf_uclear_port (qport);
	  qport->uuconf_zname = zport;
	  qport->uuconf_zprotocols = zprotos;
	  if (strcmp (pzsplit[0], "Direct") == 0)
	    {
	      qport->uuconf_ttype = UUCONF_PORTTYPE_DIRECT;
	      qport->uuconf_u.uuconf_sdirect.uuconf_zdevice = pzsplit[1];
	      qport->uuconf_u.uuconf_sdirect.uuconf_ibaud = ilow;
	      qport->uuconf_u.uuconf_sdirect.uuconf_fcarrier = FALSE;
	      qport->uuconf_u.uuconf_sdirect.uuconf_fhardflow = TRUE;
	      ppzdialer = NULL;
	    }
	  else if (strcmp (pzsplit[0], "TCP") == 0)
	    {
	      /* For a TCP port, the device name is taken as the TCP
		 port to use.  */
	      qport->uuconf_ttype = UUCONF_PORTTYPE_TCP;
	      qport->uuconf_ireliable
		= (UUCONF_RELIABLE_ENDTOEND | UUCONF_RELIABLE_RELIABLE
		   | UUCONF_RELIABLE_EIGHT | UUCONF_RELIABLE_FULLDUPLEX
		   | UUCONF_RELIABLE_SPECIFIED);
	      qport->uuconf_u.uuconf_stcp.uuconf_zport = pzsplit[1];
	      qport->uuconf_u.uuconf_stcp.uuconf_iversion = 0;
	      ppzdialer = &qport->uuconf_u.uuconf_stcp.uuconf_pzdialer;
	    }
	  else if (ctoks >= 5
		   && (strcmp (pzsplit[4], "TLI") == 0
		       || strcmp (pzsplit[4], "TLIS") == 0))
	    {
	      qport->uuconf_ttype = UUCONF_PORTTYPE_TLI;
	      qport->uuconf_u.uuconf_stli.uuconf_zdevice = pzsplit[1];
	      qport->uuconf_u.uuconf_stli.uuconf_fstream
		= strcmp (pzsplit[4], "TLIS") == 0;
	      qport->uuconf_u.uuconf_stli.uuconf_pzpush = NULL;
	      qport->uuconf_u.uuconf_stli.uuconf_zservaddr = NULL;
	      qport->uuconf_ireliable
		= (UUCONF_RELIABLE_ENDTOEND | UUCONF_RELIABLE_RELIABLE
		   | UUCONF_RELIABLE_EIGHT | UUCONF_RELIABLE_FULLDUPLEX
		   | UUCONF_RELIABLE_SPECIFIED);
	      ppzdialer = &qport->uuconf_u.uuconf_stli.uuconf_pzdialer;
	    }
	  else
	    {
	      qport->uuconf_ttype = UUCONF_PORTTYPE_MODEM;
	      qport->uuconf_u.uuconf_smodem.uuconf_zdevice = pzsplit[1];
	      if (strcmp (pzsplit[2], "-") != 0)
		qport->uuconf_u.uuconf_smodem.uuconf_zdial_device =
		  pzsplit[2];
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
	      qport->uuconf_u.uuconf_smodem.uuconf_qdialer = NULL;
	      ppzdialer = &qport->uuconf_u.uuconf_smodem.uuconf_pzdialer;
	    }

	  if (ppzdialer != NULL)
	    {
	      if (ctoks < 5)
		*ppzdialer = NULL;
	      else
		{
		  size_t c;
		  char **pzd;

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

		  *ppzdialer = pzd;
		}
	    }

	  if (pifn != NULL)
	    {
	      iret = (*pifn) (qport, pinfo);
	      if (iret != UUCONF_SUCCESS)
		{
		  if (zport != pzsplit[0])
		    free ((pointer) zport);
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

	  if (uuconf_add_block (pblock, zline) != 0
	      || (zport != pzsplit[0]
		  && uuconf_add_block (pblock, zport) != 0))
	    {
	      qglobal->ierrno = errno;
	      uuconf_free_block (pblock);
	      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	      break;
	    }
	  zline = NULL;

	  qport->uuconf_palloc = pblock;

	  iret = UUCONF_SUCCESS;

	  break;
	}

      (void) fclose (e);

      if (iret != UUCONF_NOT_FOUND)
	break;
    }

  if (zline != NULL)
    free ((pointer) zline);
  if (pzsplit != NULL)
    free ((pointer) pzsplit);

  if (iret != UUCONF_SUCCESS && iret != UUCONF_NOT_FOUND)
    {
      qglobal->zfilename = *pz;
      iret |= UUCONF_ERROR_FILENAME | UUCONF_ERROR_LINENO;
    }

  return iret;
}
