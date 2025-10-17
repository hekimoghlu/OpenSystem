/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
 * sslPriv.h - Misc. private SSL typedefs
 */

#ifndef	_SSL_PRIV_H_
#define _SSL_PRIV_H_	1

#include "sslBuildFlags.h"
#include "SecureTransportPriv.h"
#include "sslTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Diffie-Hellman support */
#define APPLE_DH		1


/*
 * Clients see an opaque SSLContextRef; internal code uses the 
 * following typedef.
 */
typedef struct SSLContext SSLContext;

#ifdef __cplusplus
}
#endif

#endif	/* _SSL_PRIV_H */
