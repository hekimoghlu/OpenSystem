/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

#ifndef __roken_rename_h__
#define __roken_rename_h__

/*
 * Libroken routines that are added libkafs
 */

#define _resolve_debug _kafs_resolve_debug

#define rk_dns_free_data _kafs_dns_free_data
#define rk_dns_lookup _kafs_dns_lookup
#define rk_dns_string_to_type _kafs_dns_string_to_type
#define rk_dns_type_to_string _kafs_dns_type_to_string
#define rk_dns_srv_order _kafs_dns_srv_order
#define rk_dns_make_query _kafs_dns_make_query
#define rk_dns_free_query _kafs_dns_free_query
#define rk_dns_parse_reply _kafs_dns_parse_reply

#ifndef HAVE_STRTOK_R
#define rk_strtok_r _kafs_strtok_r
#endif
#ifndef HAVE_STRLCPY
#define rk_strlcpy _kafs_strlcpy
#endif
#ifndef HAVE_STRSEP
#define rk_strsep _kafs_strsep
#endif

#endif /* __roken_rename_h__ */
