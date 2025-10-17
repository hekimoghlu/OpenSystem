/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
 * Printer database lookup routines
 * Copyright (c) 1989 by NeXT, Inc. 
 */

#ifndef _PRDB_H_
#define _PRDB_H_

typedef struct prdb_property {
	char *pp_key;
	char *pp_value;
} prdb_property;

typedef struct prdb_ent {
	char **pe_name;
	unsigned pe_nprops;
	prdb_property *pe_prop;
} prdb_ent;

#include <sys/cdefs.h>

__BEGIN_DECLS

void prdb_set __P((const char *));
const prdb_ent *prdb_get __P((void));
const prdb_ent *prdb_getbyname __P((const char *));
void prdb_end __P((void));

__END_DECLS

#endif /* !_PRDB_H_ */
