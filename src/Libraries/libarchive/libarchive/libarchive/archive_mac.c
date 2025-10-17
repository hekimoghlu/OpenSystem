/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

#include "archive.h"
#include "archive_platform.h"
#include "archive_mac.h"
#include "archive_read_private.h"

#ifdef HAVE_MAC_QUARANTINE
#include <quarantine.h>

void archive_read_get_quarantine_from_fd(struct archive *a, int fd)
{
	struct archive_read *ar = (struct archive_read *)a;
	qtn_file_t qf = qtn_file_alloc();
	if (qf) {
		if (!qtn_file_init_with_fd(qf, fd)) {
			ar->qf = qf;
		} else {
			qtn_file_free(qf);
		}
	}
}
#endif // HAVE_MAC_QUARANTINE
