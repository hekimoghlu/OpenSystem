/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
struct xaddr;

struct per_source_penalty;

void	srclimit_init(int, int, int, int,
    struct per_source_penalty *, const char *);
int	srclimit_check_allow(int, int);
void	srclimit_done(int);

#define SRCLIMIT_PENALTY_NONE			0
#define SRCLIMIT_PENALTY_CRASH			1
#define SRCLIMIT_PENALTY_AUTHFAIL		2
#define SRCLIMIT_PENALTY_GRACE_EXCEEDED		3
#define SRCLIMIT_PENALTY_NOAUTH			4
#define SRCLIMIT_PENALTY_REFUSECONNECTION	5

/* meaningful exit values, used by sshd listener for penalties */
#define EXIT_LOGIN_GRACE	3	/* login grace period exceeded */
#define EXIT_CHILD_CRASH	4	/* preauth child crashed */
#define EXIT_AUTH_ATTEMPTED	5	/* at least one auth attempt made */
#define EXIT_CONFIG_REFUSED	6	/* sshd_config RefuseConnection */

void	srclimit_penalise(struct xaddr *, int);
int	srclimit_penalty_check_allow(int, const char **);
void	srclimit_penalty_info(void);
