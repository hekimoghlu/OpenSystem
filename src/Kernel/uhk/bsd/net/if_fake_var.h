/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#ifndef _NET_IF_FAKE_VAR_H_
#define _NET_IF_FAKE_VAR_H_     1

#include <stdint.h>

#ifdef KERNEL_PRIVATE
__private_extern__ void
if_fake_init(void);
#endif /* KERNEL_PRIVATE */

/*
 * SIOCSDRVSPEC
 */
enum {
	IF_FAKE_S_CMD_NONE              = 0,
	IF_FAKE_S_CMD_SET_PEER          = 1,
	IF_FAKE_S_CMD_SET_MEDIA         = 2,
	IF_FAKE_S_CMD_SET_DEQUEUE_STALL = 3,
};

/*
 * SIOCGDRVSPEC
 */
enum {
	IF_FAKE_G_CMD_NONE              = 0,
	IF_FAKE_G_CMD_GET_PEER          = 1,
};

#define IF_FAKE_MEDIA_LIST_MAX  27

struct if_fake_media {
	int32_t         iffm_current;
	uint32_t        iffm_count;
	uint32_t        iffm_reserved[3];
	int32_t         iffm_list[IF_FAKE_MEDIA_LIST_MAX];
};

struct if_fake_request {
	uint64_t        iffr_reserved[4];
	union {
		char    iffru_buf[128];         /* stable size */
		struct if_fake_media    iffru_media;
		char    iffru_peer_name[IFNAMSIZ]; /* if name, e.g. "en0" */
		/*
		 * control dequeue stall. 0: disable dequeue stall, else
		 * enable dequeue stall.
		 */
		uint32_t        iffru_dequeue_stall;
	} iffr_u;
#define iffr_peer_name  iffr_u.iffru_peer_name
#define iffr_media      iffr_u.iffru_media
#define iffr_dequeue_stall      iffr_u.iffru_dequeue_stall
};

#endif /* _NET_IF_FAKE_VAR_H_ */
