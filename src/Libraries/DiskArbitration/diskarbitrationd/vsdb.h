/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#ifndef _VSDB_H_
#define _VSDB_H_

#define	_PATH_VSDB	"/var/db/volinfo.database"

#define	VSDB_PERM	0x00000001		/* enable permissions */

struct vsdb {
	char	*vs_spec;		/* volume uuid */
	int	vs_ops;		/* volume options */
};

#include <sys/cdefs.h>

__BEGIN_DECLS
struct vsdb *getvsent __P((void));
struct vsdb *getvsspec __P((const char *));
int setvsent __P((void));
void endvsent __P((void));
__END_DECLS

#endif /* !_VSDB_H_ */
