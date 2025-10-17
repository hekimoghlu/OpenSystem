/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
 */
#define	TIMEOUT		5
#define	MAX_TIMEOUTS	5

/* Generic values */
#define MAXSEGSIZE	65464		/* Maximum size of the data segment */
#define	MAXPKTSIZE	(MAXSEGSIZE + 4) /* Maximum size of the packet */

/* For the blksize option */
#define BLKSIZE_MIN	8		/* Minimum size of the data segment */
#define BLKSIZE_MAX	MAXSEGSIZE	/* Maximum size of the data segment */

/* For the timeout option */
#define TIMEOUT_MIN	0		/* Minimum timeout value */
#define TIMEOUT_MAX	255		/* Maximum timeout value */
#define MIN_TIMEOUTS	3

/* For the windowsize option */
#define	WINDOWSIZE	1
#define	WINDOWSIZE_MIN	1
#define	WINDOWSIZE_MAX	65535

extern int	timeoutpacket;
extern int	timeoutnetwork;
extern int	maxtimeouts;
int	settimeouts(int timeoutpacket, int timeoutnetwork, int maxtimeouts);

extern uint16_t	segsize;
extern uint16_t	pktsize;
extern uint16_t	windowsize;

extern int	acting_as_client;

/*
 */
void	unmappedaddr(struct sockaddr_in6 *sin6);
size_t	get_field(int peer, char *buffer, size_t size);

/*
 * Packet types
 */
struct packettypes {
	int	value;
	const char *const name;
};
extern struct packettypes packettypes[];
const char *packettype(int);

/*
 * RP_
 */
struct rp_errors {
	int	error;
	const char *const desc;
};
extern struct rp_errors rp_errors[];
char	*rp_strerror(int error);

/*
 * Debug features
 */
#define	DEBUG_NONE	0x0000
#define DEBUG_PACKETS	0x0001
#define DEBUG_SIMPLE	0x0002
#define DEBUG_OPTIONS	0x0004
#define DEBUG_ACCESS	0x0008
struct debugs {
	int	value;
	const char *const name;
	const char *const desc;
};
extern int	debug;
extern struct debugs debugs[];
extern unsigned int packetdroppercentage;
int	debug_find(char *s);
int	debug_finds(char *s);
const char *debug_show(int d);

/*
 * Log routines
 */
#define DEBUG(s) tftp_log(LOG_DEBUG, "%s", s)
extern int tftp_logtostdout;
void	tftp_openlog(const char *ident, int logopt, int facility);
void	tftp_closelog(void);
void	tftp_log(int priority, const char *message, ...) __printflike(2, 3);

/*
 * Performance figures
 */
struct tftp_stats {
	size_t		amount;
	int		rollovers;
	uint32_t	blocks;
	int		retries;
	struct timeval	tstart;
	struct timeval	tstop;
};

void	stats_init(struct tftp_stats *ts);
void	printstats(const char *direction, int verbose, struct tftp_stats *ts);
