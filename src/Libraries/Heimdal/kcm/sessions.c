/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include "kcm_locl.h"

#ifdef __APPLE__
#include <bsm/audit_kevents.h>
#include <bsm/audit_session.h>
#endif

void
kcm_session_add(pid_t session_id)
{
    kcm_log(1, "monitor session: %d\n", session_id);
}

void
kcm_session_setup_handler(void)
{
#ifdef __APPLE__
    au_sdev_handle_t *h;
    dispatch_queue_t bgq;

    h = au_sdev_open(AU_SDEVF_ALLSESSIONS);
    if (h == NULL)
	return;

    bgq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0);

    dispatch_async(bgq, ^{
	    for (;;) {
		auditinfo_addr_t aio;
		int event;
	
		if (au_sdev_read_aia(h, &event, &aio) != 0)
		    continue;

		/* 
		 * Ignore everything but END. This should relly be
		 * CLOSE but since that is delayed until the credential
		 * is reused, we can't do that 
		 * */
		if (event != AUE_SESSION_END)
		    continue;
		
		dispatch_async(dispatch_get_main_queue(), ^{
			kcm_cache_remove_session(aio.ai_asid);
		    });
	    }
	});
#endif
}
