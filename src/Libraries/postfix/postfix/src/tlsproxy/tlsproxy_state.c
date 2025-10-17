/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
  * System library.
  */
#include <sys_defs.h>

 /*
  * Utility library.
  */
#include <msg.h>
#include <mymalloc.h>
#include <nbbio.h>

 /*
  * Master library.
  */
#include <mail_server.h>

 /*
  * TLS library.
  */
#ifdef USE_TLS
#define TLS_INTERNAL			/* XXX */
#include <tls.h>

 /*
  * Application-specific.
  */
#include <tlsproxy.h>

/* tlsp_state_create - create TLS proxy state object */

TLSP_STATE *tlsp_state_create(const char *service,
			              VSTREAM *plaintext_stream)
{
    TLSP_STATE *state = (TLSP_STATE *) mymalloc(sizeof(*state));

    state->flags = TLSP_FLAG_DO_HANDSHAKE;
    state->service = mystrdup(service);
    state->plaintext_stream = plaintext_stream;
    state->plaintext_buf = 0;
    state->ciphertext_fd = -1;
    state->ciphertext_timer = 0;
    state->timeout = -1;
    state->remote_endpt = 0;
    state->server_id = 0;
    state->tls_context = 0;

    return (state);
}

/* tlsp_state_free - destroy state objects, connection and events */

void    tlsp_state_free(TLSP_STATE *state)
{
    myfree(state->service);
    if (state->plaintext_buf)			/* turns off plaintext events */
	nbbio_free(state->plaintext_buf);
    event_server_disconnect(state->plaintext_stream);
    if (state->ciphertext_fd >= 0) {
	event_disable_readwrite(state->ciphertext_fd);
	(void) close(state->ciphertext_fd);
    }
    if (state->ciphertext_timer)
	event_cancel_timer(state->ciphertext_timer, (void *) state);
    if (state->remote_endpt) {
	msg_info("DISCONNECT %s", state->remote_endpt);
	myfree(state->remote_endpt);
    }
    if (state->server_id)
	myfree(state->server_id);
    if (state->tls_context)
	tls_free_context(state->tls_context);
    myfree((void *) state);
}

#endif
