/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
 * Copyright (c) 2017 Markus Friedl.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SSHKEY_XMSS_H
#define SSHKEY_XMSS_H

#define XMSS_SHA2_256_W16_H10_NAME	"XMSS_SHA2-256_W16_H10"
#define XMSS_SHA2_256_W16_H16_NAME	"XMSS_SHA2-256_W16_H16"
#define XMSS_SHA2_256_W16_H20_NAME	"XMSS_SHA2-256_W16_H20"
#define XMSS_DEFAULT_NAME		XMSS_SHA2_256_W16_H10_NAME

size_t	 sshkey_xmss_pklen(const struct sshkey *);
size_t	 sshkey_xmss_sklen(const struct sshkey *);
int	 sshkey_xmss_init(struct sshkey *, const char *);
void	 sshkey_xmss_free_state(struct sshkey *);
int	 sshkey_xmss_generate_private_key(struct sshkey *, int);
int	 sshkey_xmss_serialize_state(const struct sshkey *, struct sshbuf *);
int	 sshkey_xmss_serialize_state_opt(const struct sshkey *, struct sshbuf *,
	    enum sshkey_serialize_rep);
int	 sshkey_xmss_serialize_pk_info(const struct sshkey *, struct sshbuf *,
	    enum sshkey_serialize_rep);
int	 sshkey_xmss_deserialize_state(struct sshkey *, struct sshbuf *);
int	 sshkey_xmss_deserialize_state_opt(struct sshkey *, struct sshbuf *);
int	 sshkey_xmss_deserialize_pk_info(struct sshkey *, struct sshbuf *);

int	 sshkey_xmss_siglen(const struct sshkey *, size_t *);
void	*sshkey_xmss_params(const struct sshkey *);
void	*sshkey_xmss_bds_state(const struct sshkey *);
int	 sshkey_xmss_get_state(const struct sshkey *, int);
int	 sshkey_xmss_enable_maxsign(struct sshkey *, u_int32_t);
int	 sshkey_xmss_forward_state(const struct sshkey *, u_int32_t);
int	 sshkey_xmss_update_state(const struct sshkey *, int);
u_int32_t sshkey_xmss_signatures_left(const struct sshkey *);

#endif /* SSHKEY_XMSS_H */
