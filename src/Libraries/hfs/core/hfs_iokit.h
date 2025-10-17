/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#ifndef hfs_iokit_h
#define hfs_iokit_h

#include <sys/cdefs.h>
#include <AppleKeyStore/AppleKeyStoreFSServices.h>

__BEGIN_DECLS

int hfs_is_ejectable(const char *cdev_name);
void hfs_iterate_media_with_content(const char *content_uuid_cstring,
									int (*func)(const char *bsd_name,
												const char *uuid_str,
												void *arg),
									void *arg);
kern_return_t hfs_get_platform_serial_number(char *serial_number_str,
											 uint32_t len);
int hfs_unwrap_key(aks_cred_t access, const aks_wrapped_key_t wrapped_key_in,
				   aks_raw_key_t key_out);
int hfs_rewrap_key(aks_cred_t access, cp_key_class_t dp_class,
				   const aks_wrapped_key_t wrapped_key_in,
				   aks_wrapped_key_t wrapped_key_out);
int hfs_new_key(aks_cred_t access, cp_key_class_t dp_class,
				aks_raw_key_t key_out, aks_wrapped_key_t wrapped_key_out);
int hfs_backup_key(aks_cred_t access, const aks_wrapped_key_t wrapped_key_in,
				   aks_wrapped_key_t wrapped_key_out);

__END_DECLS

#endif /* hfs_iokit_h */
