/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#ifndef __PPP_COMP_H__
#define __PPP_COMP_H__


int ppp_comp_init(void);
int ppp_comp_dispose(void);
void ppp_comp_alloc(struct ppp_if *wan);
void ppp_comp_dealloc(struct ppp_if *wan);
int ppp_comp_setcompressor(struct ppp_if *wan, struct ppp_option_data *odp);
void ppp_comp_getstats(struct ppp_if *wan, struct ppp_comp_stats *stats);
void ppp_comp_ccp(struct ppp_if *wan, mbuf_t m, int rcvd);
void ppp_comp_close(struct ppp_if *wan);
int ppp_comp_compress(struct ppp_if *wan, mbuf_t *m);
int ppp_comp_incompress(struct ppp_if *wan, mbuf_t m);
int ppp_comp_decompress(struct ppp_if *wan, mbuf_t *m);


#endif
