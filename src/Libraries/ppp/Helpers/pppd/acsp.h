/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#ifndef __ACSP_H__
#define __ACSP_H__

// ACSP payload dumping macros... for use within acsp_data_printpkt only
#define ACSP_PRINT_PAYLOAD(payload_name, flag_names...)															\
	(payload_name)?																								\
		printer(arg, " <payload len %d, packet seq %d, %s, flags:%s%s%s%s>",  len, pkt->seq, payload_name, ## flag_names) :	\
		printer(arg, " <payload len %d, packet seq %d, CI_TYPE %d, flags:%s%s%s%s>", len, pkt->seq, pkt->type, ## flag_names)

#define ACSP_PRINTPKT_PAYLOAD(payload_name)												\
	ACSP_PRINT_PAYLOAD(payload_name,													\
						((flags & ACSP_FLAG_START) != 0)? " START" : "",				\
						((flags & ACSP_FLAG_END) != 0)? " END" : "",					\
						((flags & ACSP_FLAG_REQUIRE_ACK) != 0)? " REQUIRE-ACK" : "",	\
						((flags & ACSP_FLAG_ACK) != 0)? " ACK" : "")

//
// ACSP function prototypes
//
void acsp_start(int mtu);
void acsp_stop(void);
void acsp_init_plugins(void *arg, uintptr_t phase);

void acsp_data_input(int unit, u_char *pkt, int len);
int acsp_printpkt(u_char *, int, void (*) __P((void *, char *, ...)), void *);


#endif