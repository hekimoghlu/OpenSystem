/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
extern int debug;
extern char hostname[];

int	announce(CTL_MSG *, const char *);
int	delete_invite(u_int32_t);
void	do_announce(CTL_MSG *, CTL_RESPONSE *);
CTL_MSG	*find_match(CTL_MSG *request);
CTL_MSG	*find_request(CTL_MSG *request);
int	find_user(const char *name, char *tty);
void	insert_table(CTL_MSG *, CTL_RESPONSE *);
int	new_id(void);
int	print_mesg(const char *, CTL_MSG *, const char *);
void	print_request(const char *, CTL_MSG *);
void	print_response(const char *, CTL_RESPONSE *);
void	process_request(CTL_MSG *mp, CTL_RESPONSE *rp);
void	timeout(int sig);
