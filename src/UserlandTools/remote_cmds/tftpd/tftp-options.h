/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
 * Options
 */

void		init_options(void);
uint16_t	make_options(int peer, char *buffer, uint16_t size);
int		parse_options(int peer, char *buffer, uint16_t size);

/* Call back functions */
int	option_tsize(int peer, struct tftphdr *, int, struct stat *);
int	option_timeout(int peer);
int	option_blksize(int peer);
int	option_blksize2(int peer);
int	option_rollover(int peer);
int	option_windowsize(int peer);

extern int options_extra_enabled;
extern int options_rfc_enabled;

struct options {
	const char	*o_type;
	char		*o_request;
	char		*o_reply;
	int		(*o_handler)(int peer);
	int		rfc;
};

extern struct options	options[];
enum opt_enum {
	OPT_TSIZE = 0,
	OPT_TIMEOUT,
	OPT_BLKSIZE,
	OPT_BLKSIZE2,
	OPT_ROLLOVER,
	OPT_WINDOWSIZE,
};

#ifdef __APPLE__
void	options_clear_request(enum opt_enum);
#endif /* __APPLE__ */
int	options_set_request(enum opt_enum, const char *, ...)
	__printf0like(2, 3);
#ifdef __APPLE__
void	options_clear_reply(enum opt_enum);
#endif /* __APPLE__ */
int	options_set_reply(enum opt_enum, const char *, ...)
	__printf0like(2, 3);
