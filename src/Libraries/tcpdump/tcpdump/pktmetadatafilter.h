/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#ifndef tcpdump_pktmetadatafilter_h
#define tcpdump_pktmetadatafilter_h

struct node;
typedef struct node node_t;

struct pkt_meta_data {
	const char *itf;
	uint32_t dlt;
	const char *proc;
	const char *eproc;
	pid_t pid;
	pid_t epid;
	const char *dir;
	const char *svc;
	uint32_t flowid;
};


node_t * parse_expression(const char *);
void print_expression(node_t *);
int evaluate_expression(node_t *, struct pkt_meta_data *);
void free_expression(node_t *);

#ifdef DEBUG
void set_parse_verbose(int val);
#endif /* DEBUG */

#endif
