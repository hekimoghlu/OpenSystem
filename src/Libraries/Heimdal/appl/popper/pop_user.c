/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#include <popper.h>
RCSID("$Id$");

/*
 *  user:   Prompt for the user name at the start of a POP session
 */

int
pop_user (POP *p)
{
    strlcpy(p->user, p->pop_parm[1], sizeof(p->user));

    if (p->auth_level == AUTH_OTP) {
#ifdef OTP
	char ss[256], *s;

	if(otp_challenge (&p->otp_ctx, p->user, ss, sizeof(ss)) == 0)
	    return pop_msg(p, POP_SUCCESS, "Password %s required for %s.",
			   ss, p->user);
	s = otp_error(&p->otp_ctx);
	return pop_msg(p, POP_FAILURE, "Permission denied%s%s",
		       s ? ":" : "", s ? s : "");
#endif
    }
    if (p->auth_level == AUTH_SASL) {
	return pop_msg(p, POP_FAILURE, "Permission denied");
    }
    return pop_msg(p, POP_SUCCESS, "Password required for %s.", p->user);
}
