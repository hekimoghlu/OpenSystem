/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
 * Copyright (c) 2014 Markus Friedl.  All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef _HMAC_H
#define _HMAC_H

/* Returns the algorithm's digest length in bytes or 0 for invalid algorithm */
size_t ssh_hmac_bytes(int alg);

struct sshbuf;
struct ssh_hmac_ctx;
struct ssh_hmac_ctx *ssh_hmac_start(int alg);

/* Sets the state of the HMAC or resets the state if key == NULL */
int ssh_hmac_init(struct ssh_hmac_ctx *ctx, const void *key, size_t klen)
	__attribute__((__bounded__(__buffer__, 2, 3)));
int ssh_hmac_update(struct ssh_hmac_ctx *ctx, const void *m, size_t mlen)
	__attribute__((__bounded__(__buffer__, 2, 3)));
int ssh_hmac_update_buffer(struct ssh_hmac_ctx *ctx, const struct sshbuf *b);
int ssh_hmac_final(struct ssh_hmac_ctx *ctx, u_char *d, size_t dlen)
	__attribute__((__bounded__(__buffer__, 2, 3)));
void ssh_hmac_free(struct ssh_hmac_ctx *ctx);

#endif /* _HMAC_H */
