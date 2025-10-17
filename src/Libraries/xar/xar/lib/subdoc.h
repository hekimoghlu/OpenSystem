/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
 * 03-Apr-2005
 * DRI: Rob Braun <bbraun@synack.net>
 */

#ifndef _XAR_SUBDOC_H_
#define _XAR_SUBDOC_H_

#include "xar.h"
#include "filetree.h"

struct __xar_subdoc_t {
	struct __xar_prop_t  *props;
	struct __xar_attr_t  *attrs;
	const char *prefix;
	const char *ns;
	const char *blank1; /* filler for xar_file_t compatibility */
	const char *blank2; /* filler for xar_file_t compatibility */
	const char blank3; /* filler for xar_file_t compatibility */
	const char *name;
	struct __xar_subdoc_t *next;
	const char *value; /* a subdoc should very rarely have a value */
	xar_t x;
};

#define XAR_SUBDOC(x) ((struct __xar_subdoc_t *)(x))

int xar_subdoc_unserialize(xar_subdoc_t s, xmlTextReaderPtr reader);
void xar_subdoc_serialize(xar_subdoc_t s, xmlTextWriterPtr writer, int wrap);
void xar_subdoc_free(xar_subdoc_t s);
xar_subdoc_t xar_subdoc_find(xar_t x, const char *name);

#endif /* _XAR_SUBDOC_H_ */
