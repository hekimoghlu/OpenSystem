/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#include <config.h>
#endif

#include "ftmacros.h"

#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <pcap.h>		// for PCAP_ERRBUF_SIZE

#include "portability.h"
#include "rpcapd.h"
#include "config_params.h"	// configuration file parameters
#include "fileconf.h"
#include "rpcap-protocol.h"
#include "log.h"

//
// Parameter names.
//
#define PARAM_ACTIVECLIENT	"ActiveClient"
#define PARAM_PASSIVECLIENT	"PassiveClient"
#define PARAM_NULLAUTHPERMIT	"NullAuthPermit"

static char *skipws(char *ptr);

/*
 * Locale-independent version checks for alphabetical and alphanumerical
 * characters that also can handle being handed a char value that might
 * be negative.
 */
#define FILECONF_ISALPHA(c) \
	(((c) >= 'A' && (c) <= 'Z') || ((c) >= 'a' && (c) <= 'z'))
#define FILECONF_ISALNUM(c) \
	(FILECONF_ISALPHA(c) || ((c) >= '0' && (c) <= '9'))

void fileconf_read(void)
{
	FILE *fp;
	unsigned int num_active_clients;

	if ((fp = fopen(loadfile, "r")) != NULL)
	{
		char line[MAX_LINE + 1];
		unsigned int lineno;

		hostlist[0] = 0;
		num_active_clients = 0;
		lineno = 0;

		while (fgets(line, MAX_LINE, fp) != NULL)
		{
			size_t linelen;
			char *ptr;
			char *param;
			size_t result;
			size_t toklen;

			lineno++;

			linelen = strlen(line);
			if (line[linelen - 1] != '\n')
			{
				int c;

				//
				// Either the line doesn't fit in
				// the buffer, or we got an EOF
				// before the EOL.  Assume it's the
				// former.
				//
				rpcapd_log(LOGPRIO_ERROR,
				    "%s, line %u is longer than %u characters",
				    loadfile, lineno, MAX_LINE);

				//
				// Eat characters until we get an NL.
				//
				while ((c = getc(fp)) != '\n')
				{
					if (c == EOF)
						goto done;
				}

				//
				// Try the next line.
				//
				continue;
			}
			ptr = line;

			//
			// Skip leading white space, if any.
			//
			ptr = skipws(ptr);
			if (ptr == NULL)
			{
				// Blank line.
				continue;
			}

			//
			// Is the next character a "#"?  If so, this
			// line is a comment; skip to the next line.
			//
			if (*ptr == '#')
				continue;

			//
			// Is the next character alphabetic?  If not,
			// this isn't a valid parameter name.
			//
			if (FILECONF_ISALPHA(*ptr))
			{
				rpcapd_log(LOGPRIO_ERROR,
				    "%s, line %u doesn't have a valid parameter name",
				    loadfile, lineno);
				continue;
			}

			//
			// Grab the first token, which is made of
			// alphanumerics, underscores, and hyphens.
			// That's the name of the parameter being set.
			//
			param = ptr;
			while (FILECONF_ISALNUM(*ptr) || *ptr == '-' || *ptr == '_')
				ptr++;

			//
			// Skip over white space, if any.
			//
			ptr = skipws(ptr);
			if (ptr == NULL || *ptr != '=')
			{
				//
				// We hit the end of the line before
				// finding a non-white space character,
				// or we found one but it's not an "=".
				// That means there's no "=", so this
				// line is invalid.  Complain and skip
				// this line.
				//
				rpcapd_log(LOGPRIO_ERROR,
				    "%s, line %u has a parameter but no =",
				    loadfile, lineno);
				continue;
			}

			//
			// We found the '='; set it to '\0', and skip
			// past it.
			//
			*ptr++ = '\0';

			//
			// Skip past any white space after the "=".
			//
			ptr = skipws(ptr);
			if (ptr == NULL)
			{
				//
				// The value is empty.
				//
				rpcapd_log(LOGPRIO_ERROR,
				    "%s, line %u has a parameter but no value",
				    loadfile, lineno);
				continue;
			}

			//
			// OK, what parameter is this?
			//
			if (strcmp(param, PARAM_ACTIVECLIENT) == 0) {
				//
				// Add this to the list of active clients.
				//
				char *address, *port;

				//
				// We can't have more than MAX_ACTIVE_LIST
				// active clients.
				//
				if (num_active_clients >= MAX_ACTIVE_LIST)
				{
					//
					// Too many entries for the active
					// client list.  Complain and
					// ignore it.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has an %s parameter, but we already have %u active clients",
					    loadfile, lineno, PARAM_ACTIVECLIENT,
					    MAX_ACTIVE_LIST);
					continue;
				}

				//
				// Get the address.
				// It's terminated by a host list separator
				// *or* a #; there *shouldn't* be a #, as
				// that starts a comment, and that would
				// mean that we have no port.
				//
				address = ptr;
				toklen = strcspn(ptr, RPCAP_HOSTLIST_SEP "#");
				ptr += toklen;	// skip to the terminator
				if (toklen == 0)
				{
					if (*ptr == ' ' || *ptr == '\t' ||
					    *ptr == '\r' || *ptr == '\n' ||
					    *ptr == '#' || *ptr == '\0')
					{
						//
						// The first character it saw
						// was a whitespace character
						// or a comment character,
						// or we ran out of characters.
						// This means that there's
						// no value.
						//
						rpcapd_log(LOGPRIO_ERROR,
						    "%s, line %u has a parameter but no value",
						    loadfile, lineno);
					}
					else
					{
						//
						// This means that the first
						// character it saw was a
						// separator.  This means that
						// there's no address in the
						// value, just a port.
						//
						rpcapd_log(LOGPRIO_ERROR,
						    "%s, line %u has an %s parameter with a value containing no address",
						    loadfile, lineno, PARAM_ACTIVECLIENT);
					}
					continue;
				}

				//
				// Null-terminate the address, and skip past
				// it.
				//
				*ptr++ = '\0';

				//
				// Skip any white space following the
				// separating character.
				//
				ptr = skipws(ptr);
				if (ptr == NULL)
				{
					//
					// The value is empty, so there's
					// no port in the value.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has an %s parameter with a value containing no port",
					    loadfile, lineno, PARAM_ACTIVECLIENT);
					continue;
				}

				//
				// Get the port.
				// We look for a white space character
				// or a # as a terminator; the # introduces
				// a comment that runs to the end of the
				// line.
				//
				port = ptr;
				toklen = strcspn(ptr, " \t#\r\n");
				ptr += toklen;
				if (toklen == 0)
				{
					//
					// The value is empty, so there's
					// no port in the value.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has an %s parameter with a value containing no port",
					    loadfile, lineno, PARAM_ACTIVECLIENT);
					continue;
				}

				//
				// Null-terminate the port, and skip past
				// it.
				//
				*ptr++ = '\0';
				result = pcap_strlcpy(activelist[num_active_clients].address, address, sizeof(activelist[num_active_clients].address));
				if (result >= sizeof(activelist[num_active_clients].address))
				{
					//
					// It didn't fit.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has an %s parameter with an address with more than %u characters",
					    loadfile, lineno, PARAM_ACTIVECLIENT,
					    (unsigned int)(sizeof(activelist[num_active_clients].address) - 1));
					continue;
				}
				if (strcmp(port, "DEFAULT") == 0) // the user choose a custom port
					result = pcap_strlcpy(activelist[num_active_clients].port, RPCAP_DEFAULT_NETPORT_ACTIVE, sizeof(activelist[num_active_clients].port));
				else
					result = pcap_strlcpy(activelist[num_active_clients].port, port, sizeof(activelist[num_active_clients].port));
				if (result >= sizeof(activelist[num_active_clients].address))
				{
					//
					// It didn't fit.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has an %s parameter with an port with more than %u characters",
					    loadfile, lineno, PARAM_ACTIVECLIENT,
					    (unsigned int)(sizeof(activelist[num_active_clients].port) - 1));
					continue;
				}

				num_active_clients++;
			}
			else if (strcmp(param, PARAM_PASSIVECLIENT) == 0)
			{
				char *eos;
				char *host;

				//
				// Get the host.
				// We look for a white space character
				// or a # as a terminator; the # introduces
				// a comment that runs to the end of the
				// line.
				//
				host = ptr;
				toklen = strcspn(ptr, " \t#\r\n");
				if (toklen == 0)
				{
					//
					// The first character it saw
					// was a whitespace character
					// or a comment character.
					// This means that there's
					// no value.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has a parameter but no value",
					    loadfile, lineno);
					continue;
				}
				ptr += toklen;
				*ptr++ = '\0';

				//
				// Append this to the host list.
				// Save the curren end-of-string for the
				// host list, in case the new host doesn't
				// fit, so that we can discard the partially-
				// copied host name.
				//
				eos = hostlist + strlen(hostlist);
				if (eos != hostlist)
				{
					//
					// The list is not empty, so prepend
					// a comma before adding this host.
					//
					result = pcap_strlcat(hostlist, ",", sizeof(hostlist));
					if (result >= sizeof(hostlist))
					{
						//
						// It didn't fit.  Discard
						// the comma (which wasn't
						// added, but...), complain,
						// and ignore this line.
						//
						*eos = '\0';
						rpcapd_log(LOGPRIO_ERROR,
						    "%s, line %u has a %s parameter with a host name that doesn't fit",
						    loadfile, lineno, PARAM_PASSIVECLIENT);
						continue;
					}
				}
				result = pcap_strlcat(hostlist, host, sizeof(hostlist));
				if (result >= sizeof(hostlist))
				{
					//
					// It didn't fit.  Discard the comma,
					// complain, and ignore this line.
					//
					*eos = '\0';
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has a %s parameter with a host name that doesn't fit",
					    loadfile, lineno, PARAM_PASSIVECLIENT);
					continue;
				}
			}
			else if (strcmp(param, PARAM_NULLAUTHPERMIT) == 0)
			{
				char *setting;

				//
				// Get the setting.
				// We look for a white space character
				// or a # as a terminator; the # introduces
				// a comment that runs to the end of the
				// line.
				//
				setting = ptr;
				toklen = strcspn(ptr, " \t#\r\n");
				ptr += toklen;
				if (toklen == 0)
				{
					//
					// The first character it saw
					// was a whitespace character
					// or a comment character.
					// This means that there's
					// no value.
					//
					rpcapd_log(LOGPRIO_ERROR,
					    "%s, line %u has a parameter but no value",
					    loadfile, lineno);
					continue;
				}
				*ptr++ = '\0';

				//
				// XXX - should we complain if it's
				// neither "yes" nor "no"?
				//
				if (strcmp(setting, "YES") == 0)
					nullAuthAllowed = 1;
				else
					nullAuthAllowed = 0;
			}
			else
			{
				rpcapd_log(LOGPRIO_ERROR,
				    "%s, line %u has an unknown parameter %s",
				    loadfile, lineno, param);
				continue;
			}
		}

done:
		// clear the remaining fields of the active list
		for (int i = num_active_clients; i < MAX_ACTIVE_LIST; i++)
		{
			activelist[i].address[0] = 0;
			activelist[i].port[0] = 0;
			num_active_clients++;
		}

		rpcapd_log(LOGPRIO_DEBUG, "New passive host list: %s", hostlist);
		fclose(fp);
	}
}

int fileconf_save(const char *savefile)
{
	FILE *fp;

	if ((fp = fopen(savefile, "w")) != NULL)
	{
		char *token; /*, *port;*/					// temp, needed to separate items into the hostlist
		char temphostlist[MAX_HOST_LIST + 1];
		int i = 0;
		char *lasts;

		fprintf(fp, "# Configuration file help.\n\n");

		// Save list of clients which are allowed to connect to us in passive mode
		fprintf(fp, "# Hosts which are allowed to connect to this server (passive mode)\n");
		fprintf(fp, "# Format: PassiveClient = <name or address>\n\n");

		pcap_strlcpy(temphostlist, hostlist, sizeof (temphostlist));

		token = pcap_strtok_r(temphostlist, RPCAP_HOSTLIST_SEP, &lasts);
		while(token != NULL)
		{
			fprintf(fp, "%s = %s\n", PARAM_PASSIVECLIENT, token);
			token = pcap_strtok_r(NULL, RPCAP_HOSTLIST_SEP, &lasts);
		}


		// Save list of clients which are allowed to connect to us in active mode
		fprintf(fp, "\n\n");
		fprintf(fp, "# Hosts to which this server is trying to connect to (active mode)\n");
		fprintf(fp, "# Format: ActiveClient = <name or address>, <port | DEFAULT>\n\n");


		while ((i < MAX_ACTIVE_LIST) && (activelist[i].address[0] != 0))
		{
			fprintf(fp, "%s = %s, %s\n", PARAM_ACTIVECLIENT,
			    activelist[i].address, activelist[i].port);
			i++;
		}

		// Save if we want to permit NULL authentication
		fprintf(fp, "\n\n");
		fprintf(fp, "# Permit NULL authentication: YES or NO\n\n");

		fprintf(fp, "%s = %s\n", PARAM_NULLAUTHPERMIT,
		    nullAuthAllowed ? "YES" : "NO");

		fclose(fp);
		return 0;
	}
	else
	{
		return -1;
	}

}

//
// Skip over white space.
// If we hit a CR or LF, return NULL, otherwise return a pointer to
// the first non-white space character.  Replace white space characters
// other than CR or LF with '\0', so that, if we're skipping white space
// after a token, the token is null-terminated.
//
static char *skipws(char *ptr)
{
	while (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n') {
		if (*ptr == '\r' || *ptr == '\n')
			return NULL;
		*ptr++ = '\0';
	}
	return ptr;
}
