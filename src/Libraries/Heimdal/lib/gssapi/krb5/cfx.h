/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
/* $Id$ */

#ifndef GSSAPI_CFX_H_
#define GSSAPI_CFX_H_ 1

/*
 * Implementation of draft-ietf-krb-wg-gssapi-cfx-01.txt
 */

typedef struct gss_cfx_mic_token_desc_struct {
	u_char TOK_ID[2]; /* 04 04 */
	u_char Flags;
	u_char Filler[5];
	u_char SND_SEQ[8];
} gss_cfx_mic_token_desc, *gss_cfx_mic_token;

typedef struct gss_cfx_wrap_token_desc_struct {
	u_char TOK_ID[2]; /* 04 05 */
	u_char Flags;
	u_char Filler;
	u_char EC[2];
	u_char RRC[2];
	u_char SND_SEQ[8];
} gss_cfx_wrap_token_desc, *gss_cfx_wrap_token;

typedef struct gss_cfx_delete_token_desc_struct {
	u_char TOK_ID[2]; /* 05 04 */
	u_char Flags;
	u_char Filler[5];
	u_char SND_SEQ[8];
} gss_cfx_delete_token_desc, *gss_cfx_delete_token;

#endif /* GSSAPI_CFX_H_ */
