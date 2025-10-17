/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
 * Helpers for key API tests
 *
 * Placed in the public domain
 */

/* Load a binary file into a buffer */
struct sshbuf *load_file(const char *name);

/* Load a text file into a buffer */
struct sshbuf *load_text_file(const char *name);

/* Load a bignum from a file */
BIGNUM *load_bignum(const char *name);

/* Accessors for key components */
const BIGNUM *rsa_n(struct sshkey *k);
const BIGNUM *rsa_e(struct sshkey *k);
const BIGNUM *rsa_p(struct sshkey *k);
const BIGNUM *rsa_q(struct sshkey *k);
const BIGNUM *dsa_g(struct sshkey *k);
const BIGNUM *dsa_pub_key(struct sshkey *k);
const BIGNUM *dsa_priv_key(struct sshkey *k);

