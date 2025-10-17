/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
 * Mail alias lookup routines
 * Copyright (c) 1989 by NeXT, Inc.
 */

#ifndef _ALIAS_H_
#define _ALIAS_H_

struct aliasent {
	char		*alias_name;
	unsigned	alias_members_len;
	char		**alias_members;
	int			alias_local;
};

#include <sys/cdefs.h>

__BEGIN_DECLS
void alias_setent __P((void));
struct aliasent *alias_getent __P((void));
void alias_endent __P((void));
struct aliasent *alias_getbyname __P((const char *));
__END_DECLS

#endif /* !_ALIAS_H_ */
