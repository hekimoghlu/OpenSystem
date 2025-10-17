/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#ifndef _S_AFPUSERS_H
#define _S_AFPUSERS_H

#include <unistd.h>
#include <stdint.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFDictionary.h>
#include <OpenDirectory/OpenDirectory.h>
#include <DirectoryService/DirectoryService.h>

#define CHARSET_SYMBOLS			"-,./[]\\;'!@#%&*()_{}:\"?"
#define CHARSET_SYMBOLS_LENGTH		(sizeof(CHARSET_SYMBOLS) - 1)

typedef CFMutableDictionaryRef	AFPUserRef;

typedef struct {
    ODNodeRef		node;
    CFMutableArrayRef	list;
    ODRecordRef		afp_access_group;
} AFPUserList, *AFPUserListRef;

void		AFPUserList_free(AFPUserListRef users);
Boolean		AFPUserList_init(AFPUserListRef users);
Boolean		AFPUserList_create(AFPUserListRef users, gid_t gid,
				   uid_t start, int count);
AFPUserRef	AFPUserList_lookup(AFPUserListRef users, CFStringRef afp_user);

uid_t		AFPUser_get_uid(AFPUserRef user);
char *		AFPUser_get_user(AFPUserRef user, char *buf, size_t buf_len);
Boolean		AFPUser_set_random_password(AFPUserRef user, 
					    char * passwd, size_t passwd_len);

#endif	// _S_AFPUSERS_H
