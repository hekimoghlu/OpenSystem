/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#ifndef _ATOM_H
#define	_ATOM_H

#ifdef __cplusplus
extern "C" {
#endif

#define ATOM_NULL ((atom_t *)NULL)

typedef const struct atom atom_t;

atom_t *atom_get(const char *s);
atom_t *atom_get_consume(char *s);

unsigned atom_hash(atom_t *atom);

#ifndef __cplusplus
struct atom {
	char value[0];
};

static inline const char *
atom_pretty(atom_t *atom, const char *nullstr)
{
	return atom ? atom->value : nullstr;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* _ATOM_H */
