/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
 * Copyright (C) 1995, 1996, 1997, and 1998 WIDE Project.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _ALGORITHM_TYPES_H
#define _ALGORITHM_TYPES_H


/* algorithm class */
enum {
	algclass_ipsec_enc,
	algclass_ipsec_auth,
	algclass_ipsec_comp,
	algclass_isakmp_enc,
	algclass_isakmp_hash,
	algclass_isakmp_dh,
	algclass_isakmp_ameth,	/* authentication method. */
#define MAXALGCLASS 7
};

#define ALG_DEFAULT_KEYLEN	64

#define ALGTYPE_NOTHING		0

/* algorithm type */
enum algtype {
	algtype_nothing = 0,

	/* enc */
	algtype_des_iv64,
	algtype_des,
	algtype_3des,
	algtype_rc5,
	algtype_idea,
	algtype_cast128,
	algtype_blowfish,
	algtype_3idea,
	algtype_des_iv32,
	algtype_rc4,
	algtype_null_enc,
	algtype_aes,
	algtype_twofish,

	/* ipsec auth */
	algtype_hmac_md5_128,
	algtype_hmac_sha1_160,
	algtype_des_mac,
	algtype_kpdk,
	algtype_non_auth,
	algtype_hmac_sha2_256,
	algtype_hmac_sha2_384,
	algtype_hmac_sha2_512,
    algtype_hmac_md5_96,
    algtype_hmac_sha1_96,

	/* ipcomp */
	algtype_oui,
	algtype_deflate,
	algtype_lzs,

	/* hash */
	algtype_md5,
	algtype_sha1,
	algtype_tiger,
	algtype_sha2_256,
	algtype_sha2_384,
	algtype_sha2_512,

	/* dh_group */
	algtype_modp768,
	algtype_modp1024,
	algtype_ec2n155,
	algtype_ec2n185,
	algtype_modp1536,
	algtype_modp2048,
	algtype_modp3072,
	algtype_modp4096,
	algtype_modp6144,
	algtype_modp8192,

	/* authentication method. */
	algtype_psk,
	algtype_dsssig,
	algtype_rsasig,
	algtype_rsaenc,
	algtype_rsarev,
	algtype_gssapikrb,
#ifdef ENABLE_HYBRID
	algtype_hybrid_rsa_s,
	algtype_hybrid_dss_s,
	algtype_hybrid_rsa_c,
	algtype_hybrid_dss_c,
	algtype_xauth_psk_s,
	algtype_xauth_psk_c,
	algtype_xauth_rsa_s,
	algtype_xauth_rsa_c,
    algtype_eap_psk_c,
    algtype_eap_rsa_c,
#endif
};

#endif