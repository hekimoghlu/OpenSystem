/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#ifndef _LIBUTIL_H_
#define	_LIBUTIL_H_

#include <unistd.h>
#include <stdbool.h>

#ifdef _SYS_PARAM_H_
/* for pidfile.c */
struct pidfh {
	int	pf_fd;
	char	pf_path[MAXPATHLEN + 1];
	dev_t	pf_dev;
	ino_t	pf_ino;
};
#else
struct pidfh;
#endif

struct in_addr;
struct sockaddr;

__BEGIN_DECLS
int	expand_number(const char *_buf, uint64_t *_num);
int	humanize_number(char *_buf, size_t _len, int64_t _number,
	    const char *_suffix, int _scale, int _flags);

int	realhostname(char *host, size_t hsize, const struct in_addr *ip);
int	realhostname_sa(char *host, size_t hsize, struct sockaddr *addr,
			     int addrlen);

struct pidfh *pidfile_open(const char *path, mode_t mode, pid_t *pidptr);
int pidfile_write(struct pidfh *pfh);
int pidfile_close(struct pidfh *pfh);
int pidfile_remove(struct pidfh *pfh);

int reexec_to_match_kernel(void);
int reexec_to_match_lp64ness(bool isLP64);

__END_DECLS

/* return values from realhostname() */
#define HOSTNAME_FOUND		(0)
#define HOSTNAME_INCORRECTNAME	(1)
#define HOSTNAME_INVALIDADDR	(2)
#define HOSTNAME_INVALIDNAME	(3)

/* Values for humanize_number(3)'s flags parameter. */
#define	HN_DECIMAL		0x01
#define	HN_NOSPACE		0x02
#define	HN_B			0x04
#define	HN_DIVISOR_1000		0x08
#define	HN_IEC_PREFIXES		0x10

/* Values for humanize_number(3)'s scale parameter. */
#define	HN_GETSCALE		0x10
#define	HN_AUTOSCALE		0x20

#endif /* !_LIBUTIL_H_ */
