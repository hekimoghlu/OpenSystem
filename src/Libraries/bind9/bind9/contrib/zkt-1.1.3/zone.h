/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
#ifndef ZONE_H
# define ZONE_H

# include <sys/types.h>
# include <stdio.h>
# include <time.h>
# include "dki.h"

/* all we have to know about a zone */
typedef	struct	Zone {
	const	char	*zone;	/* domain name or label */
	const	char	*dir;	/* directory of zone data */
	const	char	*file;	/* file name (zone.db)  */
	const	char	*sfile;	/* file name of secured zone (zone.db.signed)  */
	const	zconf_t	*conf;	/* ptr to config */	/* TODO: Should this be only a ptr to a local config ? */
		dki_t	*keys;	/* ptr to keylist */
	struct	Zone	*next;		/* ptr to next entry in list */
} zone_t;

extern	void	zone_free (zone_t *zp);
extern	void	zone_freelist (zone_t **listp);
extern	zone_t	*zone_new (zone_t **zp, const char *zone, const char *dir, const char *file, const char *signed_ext, const zconf_t *cp);
extern	const	char	*zone_geterrstr ();
extern	zone_t	*zone_add (zone_t **list, zone_t *new);
extern	const zone_t	*zone_search (const zone_t *list, const char *name);
extern	int	zone_readdir (const char *dir, const char *zone, const char *zfile, zone_t **listp, const zconf_t *conf, int dyn_zone);
extern	const	char	*zone_geterrstr (void);
extern	int	zone_print (const char *mesg, const zone_t *z);

#endif
