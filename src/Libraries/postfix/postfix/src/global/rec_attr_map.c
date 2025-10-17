/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

/* Global library. */

#include <mail_proto.h>
#include <rec_type.h>
#include <rec_attr_map.h>

/* rec_attr_map - map named attribute to pseudo record type */

int     rec_attr_map(const char *attr_name)
{
    if (strcmp(attr_name, MAIL_ATTR_DSN_ORCPT) == 0) {
	return (REC_TYPE_DSN_ORCPT);
    } else if (strcmp(attr_name, MAIL_ATTR_DSN_NOTIFY) == 0) {
	return (REC_TYPE_DSN_NOTIFY);
    } else if (strcmp(attr_name, MAIL_ATTR_DSN_ENVID) == 0) {
	return (REC_TYPE_DSN_ENVID);
    } else if (strcmp(attr_name, MAIL_ATTR_DSN_RET) == 0) {
	return (REC_TYPE_DSN_RET);
    } else if (strcmp(attr_name, MAIL_ATTR_CREATE_TIME) == 0) {
	return (REC_TYPE_CTIME);
    } else {
	return (0);
    }
}
