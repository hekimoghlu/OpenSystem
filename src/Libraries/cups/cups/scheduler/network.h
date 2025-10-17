/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
 * Structures...
 */

typedef struct cupsd_netif_s		/**** Network interface data ****/
{
  int			is_local,	/* Local (not point-to-point) interface? */
			port;		/* Listen port */
  http_addr_t		address,	/* Network address */
			mask,		/* Network mask */
			broadcast;	/* Broadcast address */
  size_t		hostlen;	/* Length of hostname */
  char			name[32],	/* Network interface name */
			hostname[1];	/* Hostname associated with interface */
} cupsd_netif_t;


/*
 * Globals...
 */

VAR int			NetIFUpdate	VALUE(1);
					/* Network interface list needs updating */
VAR cups_array_t	*NetIFList	VALUE(NULL);
					/* Array of network interfaces */

/*
 * Prototypes...
 */

extern cupsd_netif_t	*cupsdNetIFFind(const char *name);
extern void		cupsdNetIFUpdate(void);
