/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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
 * 6-July-2006
 * DRI: Christopher Ryan <ryanc@apple.com>
*/

#ifndef _XAR_SIGNATURE_H_
#define _XAR_SIGNATURE_H_

#include "xar.h"

struct __xar_x509cert_t{
	uint8_t *content;
	int32_t	len;
	struct __xar_x509cert_t *next;
};

struct __xar_signature_t {
	char *type;
	int32_t	len;
	off_t  offset;
	int32_t x509cert_count;
	struct __xar_x509cert_t *x509certs;
	struct __xar_signature_t *next;
	xar_signer_callback signer_callback;		/* callback for signing */
	void	*callback_context;					/* context for callback */
	xar_t x;
#ifdef __APPLE__
    int is_extended;
#endif
};

#define XAR_SIGNATURE(x) ((struct __xar_signature_t *)(x))

#ifdef __APPLE__
xar_signature_t xar_signature_new_internal(xar_t x, int is_extended, const char *type, int32_t length, xar_signer_callback callback, void *callback_context);
#endif

int32_t xar_signature_serialize(xar_signature_t sig, xmlTextWriterPtr writer);
xar_signature_t xar_signature_unserialize(xar_t x, xmlTextReaderPtr reader);


/* deallocates the link list of xar signatures */
void xar_signature_remove(xar_signature_t sig);

#endif /* _XAR_SIGNATURE_H_ */
