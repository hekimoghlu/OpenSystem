/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#ifndef ROLLOVER_H
# define ROLLOVER_H
# include <sys/types.h>
# include <stdarg.h>
# include <stdio.h>

#ifndef ZCONF_H
# include "zconf.h"
#endif

# define	OFFSET			((int) (2.5 * MINSEC))
# define	PARENT_PROPAGATION	(5 * MINSEC)
# define	ADD_HOLD_DOWN		(30 * DAYSEC)
# define	REMOVE_HOLD_DOWN	(30 * DAYSEC)

extern	int	ksk5011status (dki_t **listp, const char *dir, const char *domain, const zconf_t *z);
extern	int	kskstatus (zone_t *zonelist, zone_t *zp);
extern	int	zskstatus (dki_t **listp, const char *dir, const char *domain, const zconf_t *z);
#endif
