/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#ifndef _COMIMAGE_H
#define _COMIMAGE_H 1

/* init funcs for modules */
PRIVATE void rpc__module_init_func(void);

PRIVATE void rpc__register_protseq(rpc_protseq_id_elt_p_t elt, int number);
PRIVATE void rpc__register_tower_prot_id(rpc_tower_prot_ids_p_t tower_prot, int number);
PRIVATE void rpc__register_protocol_id(rpc_protocol_id_elt_p_t prot, int number);
PRIVATE void rpc__register_naf_id(rpc_naf_id_elt_p_t naf, int number);
PRIVATE void rpc__register_authn_protocol(rpc_authn_protocol_id_elt_p_t auth, int number);

#endif
