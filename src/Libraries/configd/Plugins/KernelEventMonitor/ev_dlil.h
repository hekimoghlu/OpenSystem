/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
 * Modification History
 *
 * August 5, 2002	Allan Nathanson <ajn@apple.com>
 * - split code out from eventmon.c
 */


#ifndef _EV_DLIL_H
#define _EV_DLIL_H

__BEGIN_DECLS

void	interface_detaching		(const char *if_name);

void	interface_update_delegation	(const char *if_name);

void	interface_update_idle_state	(const char *if_name);

void	interface_update_link_issues	(const char 	*if_name,
					 uint64_t	timestamp,
					 uint8_t	*modid,
					 size_t		modid_size,
					 uint8_t	*info,
					 size_t		info_size);

void	interface_update_quality_metric	(const char *if_name, int quality);

void	link_add			(const char *if_name);

void	link_remove			(const char *if_name);

void	link_update_status		(const char *if_name, boolean_t attach, boolean_t only_if_different);

void	link_update_status_if_missing	(const char * if_name);

CFMutableArrayRef
interfaceListCopy(void);

void
interfaceListUpdate(CFArrayRef ifList);

Boolean
interfaceListAddInterface(CFMutableArrayRef ifList, const char * if_name);

__END_DECLS

#endif /* _EV_DLIL_H */

