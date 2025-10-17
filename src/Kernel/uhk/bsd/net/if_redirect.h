/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#ifndef _NET_IF_REDIRECT_VAR_H_
#define _NET_IF_REDIRECT_VAR_H_     1

#ifdef KERNEL_PRIVATE
__private_extern__ void if_redirect_init(void);
#endif /* KERNEL_PRIVATE */

/* Arbitrary identifier for params type */
#define RD_CREATE_PARAMS_TYPE 0x2D27

/*
 * This lets us create an rd interface without auto-attaching fsw. Sometimes we
 * need to be able to disable fsw auto-attachment, one example being skywalk
 * unit tests (e.g. skt_xferrdudpping): rdar://109413097
 */
#define RD_CREATE_PARAMS_TYPE_NOATTACH 0x2D28

struct if_redirect_create_params {
	uint16_t ircp_type;
	uint16_t ircp_len;
	uint32_t ircp_ftype;
};

/*
 * SIOCSDRVSPEC
 */
enum {
	RD_S_CMD_NONE              = 0,
	RD_S_CMD_SET_DELEGATE      = 1,
};

struct if_redirect_request {
	uint64_t ifrr_reserved[4];
	union {
		char ifrru_buf[128];                /* stable size */
		char ifrru_delegate_name[IFNAMSIZ]; /* if name */
	} ifrr_u;
#define ifrr_delegate_name ifrr_u.ifrru_delegate_name
};

#endif /* _NET_IF_REDIRECT_VAR_H_ */
