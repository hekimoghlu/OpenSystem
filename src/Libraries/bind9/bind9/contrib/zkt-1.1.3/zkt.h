/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#ifndef ZKT_H
# define ZKT_H

extern const	dki_t	*zkt_search (const dki_t *data, int searchtag, const char *keyname);
extern	void	zkt_list_keys (const dki_t *data);
extern	void	zkt_list_trustedkeys (const dki_t *data);
extern	void	zkt_list_managedkeys (const dki_t *data);
extern	void	zkt_list_dnskeys (const dki_t *data);
extern	void	zkt_setkeylifetime (dki_t *data);

#endif
