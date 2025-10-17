/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#ifndef _NOTIFY_SERVICE_H_
#define _NOTIFY_SERVICE_H_

#define SERVICE_TYPE_NONE 0
#define SERVICE_TYPE_PATH_PUBLIC 1
#define SERVICE_TYPE_PATH_PRIVATE 2

#define SERVICE_PREFIX "com.apple.system.notify.service."
#define SERVICE_PREFIX_LEN 32

typedef struct
{
	uint32_t type;
	void *private;
} svc_info_t;

int service_open(const char *name, client_t *client, audit_token_t audit);
int service_open_path(const char *name, const char *path, uid_t uid, gid_t gid);
int service_open_path_private(const char *name, client_t *client, const char *path, audit_token_t audit, uint32_t flags);
void service_close(uint16_t service_index);
void *service_info_get(uint16_t index);

#endif /* _NOTIFY_SERVICE_H_ */
