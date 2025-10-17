/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#ifndef UUID_BUILD_STANDALONE
#include <dce/dce.h>
#include <dce/uuid.h>           /* uuid idl definitions (public)        */
#include <dce/rpcsts.h>
#else
#include "uuid.h"
#endif

#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int main(int argc ATTRIBUTE_UNUSED, char *argv[])
{
	idl_uuid_t uuid;
	unsigned32 st;
	unsigned_char_p_t uuid_string;

	uuid_create(&uuid, &st);
	if (st != error_status_ok) {
		fprintf(stderr, "%s: failed to generate UUID\n", argv[0]);
		exit(2);
	}

	uuid_to_string(&uuid, &uuid_string, &st);
	if (st != error_status_ok) {
		fprintf(stderr, "%s: failed to convert UUID to string\n", argv[0]);
		exit(3);
	}

	printf("%s\n", uuid_string);
	free(uuid_string);

	exit(0);
	return 0;
}
