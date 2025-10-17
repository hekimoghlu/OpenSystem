/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
/*
 * Browse protocols...
 */

#define BROWSE_DNSSD	1		/* DNS Service Discovery (aka Bonjour) */
#define BROWSE_SMB	2		/* SMB/Samba */
#define BROWSE_LPD	4		/* LPD via xinetd or launchd */
#define BROWSE_ALL	7		/* All protocols */


/*
 * Globals...
 */

VAR int			Browsing	VALUE(TRUE),
					/* Whether or not browsing is enabled */
			BrowseWebIF	VALUE(FALSE),
					/* Whether the web interface is advertised */
			BrowseLocalProtocols
					VALUE(BROWSE_ALL);
					/* Protocols to support for local printers */
#if defined(HAVE_DNSSD) || defined(HAVE_AVAHI)
VAR char		*DNSSDComputerName VALUE(NULL),
					/* Computer/server name */
			*DNSSDHostName	VALUE(NULL),
					/* Hostname */
			*DNSSDSubTypes VALUE(NULL);
					/* Bonjour registration subtypes */
VAR cups_array_t	*DNSSDAlias	VALUE(NULL);
					/* List of dynamic ServerAlias's */
VAR int			DNSSDPort	VALUE(0);
					/* Port number to register */
VAR cups_array_t	*DNSSDPrinters	VALUE(NULL);
					/* Printers we have registered */
#  ifdef HAVE_DNSSD
VAR DNSServiceRef	DNSSDMaster	VALUE(NULL);
					/* Master DNS-SD service reference */
#  else /* HAVE_AVAHI */
VAR AvahiThreadedPoll	*DNSSDMaster	VALUE(NULL);
					/* Master polling interface for Avahi */
VAR AvahiClient		*DNSSDClient	VALUE(NULL);
					/* Client information */
#  endif /* HAVE_DNSSD */
VAR cupsd_srv_t		WebIFSrv	VALUE(NULL);
					/* Service reference for the web interface */
#endif /* HAVE_DNSSD || HAVE_AVAHI */

VAR char		*LPDConfigFile	VALUE(NULL),
					/* LPD configuration file */
			*SMBConfigFile	VALUE(NULL);
					/* SMB configuration file */


/*
 * Prototypes...
 */

extern void	cupsdDeregisterPrinter(cupsd_printer_t *p, int removeit);
extern void	cupsdRegisterPrinter(cupsd_printer_t *p);
extern void	cupsdStartBrowsing(void);
extern void	cupsdStopBrowsing(void);
#if defined(HAVE_DNSSD) || defined(HAVE_AVAHI)
extern void	cupsdUpdateDNSSDName(void);
#endif /* HAVE_DNSSD || HAVE_AVAHI */
