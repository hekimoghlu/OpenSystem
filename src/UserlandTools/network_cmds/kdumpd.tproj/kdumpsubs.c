/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
/* Simple minded read-ahead/write-behind subroutines for tftp user and
   server.  Written originally with multiple buffers in mind, but current
   implementation has two buffer logic wired in.

   Todo:  add some sort of final error check so when the write-buffer
   is finally flushed, the caller can detect if the disk filled up
   (or had an i/o error) and return a nak to the other side.

			Jim Guyton 10/85
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif
#include <netinet/in.h>
#include "kdump.h"

#include <stdio.h>
#include <unistd.h>
#include <syslog.h>

#include "kdumpsubs.h"

#define PKTSIZE SEGSIZE+6       /* should be moved to kdump.h */

struct bf {
	int counter;            /* size of data in buffer, or flag */
	char buf[MAXIMUM_KDP_PKTSIZE];      /* room for data packet */
} bfs[2];

				/* Values for bf.counter  */
#define BF_ALLOC -3             /* alloc'd but not yet filled */
#define BF_FREE  -2             /* free */
/* [-1 .. SEGSIZE] = size of data in the data buffer */

static int nextone;		/* index of next buffer to use */
static int current;		/* index of buffer in use */

				/* control flags for crlf conversions */
int newline = 0;		/* fillbuf: in middle of newline expansion */
int prevchar = -1;		/* putbuf: previous char (cr check) */

static struct kdumphdr *rw_init __P ((int));

struct kdumphdr *w_init() { return rw_init(0); }         /* write-behind */
struct kdumphdr *r_init() { return rw_init(1); }         /* read-ahead */

extern uint32_t kdp_crashdump_pkt_size;
extern uint32_t kdp_crashdump_seg_size;

/* init for either read-ahead or write-behind */
/* zero for write-behind, one for read-head */
static struct kdumphdr *
rw_init(int x)
{
	newline = 0;		/* init crlf flag */
	prevchar = -1;
	bfs[0].counter =  BF_ALLOC;     /* pass out the first buffer */
	current = 0;
	bfs[1].counter = BF_FREE;
	nextone = x;                    /* ahead or behind? */
	return (struct kdumphdr *)bfs[0].buf;
}


/* Have emptied current buffer by sending to net and getting ack.
   Free it and return next buffer filled with data.
 */
/* if true, convert to ascii */
/* file opened for read */

/* int */
/* readit(FILE *file, struct kdumphdr **dpp, int convert) */
/* { */
/* 	struct bf *b; */

/* 	bfs[current].counter = BF_FREE; /\* free old one *\/ */
/* 	current = !current;             /\* "incr" current *\/ */

/* 	b = &bfs[current];              /\* look at new buffer *\/ */
/* 	if (b->counter == BF_FREE)      /\* if it's empty *\/ */
/* 		read_ahead(file, convert);      /\* fill it *\/ */
/* /\*      assert(b->counter != BF_FREE);*\//\* check *\/ */
/* 	*dpp = (struct kdumphdr *)b->buf;        /\* set caller's ptr *\/ */
/* 	return b->counter; */
/* } */

/*
 * fill the input buffer, doing ascii conversions if requested
 * conversions are  lf -> cr,lf  and cr -> cr, nul
 */
/*	FILE *file;  file opened for read */
/*	int convert;  if true, convert to ascii */
void
read_ahead(FILE *file, int convert)
{
	register int i;
	register char *p;
	register int c;
	struct bf *b;
	struct kdumphdr *dp;

	b = &bfs[nextone];              /* look at "next" buffer */
	if (b->counter != BF_FREE)      /* nop if not free */
		return;
	nextone = !nextone;             /* "incr" next buffer ptr */

	dp = (struct kdumphdr *)b->buf;

	if (convert == 0) {
		b->counter = read(fileno(file), dp->th_data, kdp_crashdump_seg_size);
		return;
	}

	p = dp->th_data;
	for (i = 0 ; i < kdp_crashdump_seg_size; i++) {
		if (newline) {
			if (prevchar == '\n')
				c = '\n';       /* lf to cr,lf */
			else    c = '\0';       /* cr to cr,nul */
			newline = 0;
		}
		else {
			c = getc(file);
			if (c == EOF) break;
			if (c == '\n' || c == '\r') {
				prevchar = c;
				c = '\r';
				newline = 1;
			}
		}
	       *p++ = c;
	}
	b->counter = (int)(p - dp->th_data);
}

/* Update count associated with the buffer, get new buffer
   from the queue.  Calls write_behind only if next buffer not
   available.
 */
int
writeit(FILE *file, struct kdumphdr **dpp, int ct, int convert)
{
	bfs[current].counter = ct;      /* set size of data to write */
	current = !current;             /* switch to other buffer */
	if (bfs[current].counter != BF_FREE)     /* if not free */
		(void)write_behind(file, convert); /* flush it */
	bfs[current].counter = BF_ALLOC;        /* mark as alloc'd */
	*dpp =  (struct kdumphdr *)bfs[current].buf;
	return ct;                      /* this is a lie of course */
}


/*
 * Output a buffer to a file, converting from netascii if requested.
 * CR,NUL -> CR  and CR,LF => LF.
 * Note spec is undefined if we get CR as last byte of file or a
 * CR followed by anything else.  In this case we leave it alone.
 */
int
write_behind(FILE *file, int convert)
{
	char *buf;
	int count;
	register int ct;
	register char *p;
	register int c;                 /* current character */
	struct bf *b;
	struct kdumphdr *dp;

	b = &bfs[nextone];
	if (b->counter < -1)            /* anything to flush? */
		return 0;               /* just nop if nothing to do */

	count = b->counter;             /* remember byte count */
	b->counter = BF_FREE;           /* reset flag */
	dp = (struct kdumphdr *)b->buf;
	nextone = !nextone;             /* incr for next time */
	buf = dp->th_data;

	if (count <= 0) return -1;      /* nak logic? */

	if (convert == 0)
		return write(fileno(file), buf, count);

	p = buf;
	ct = count;
	while (ct--) {                  /* loop over the buffer */
	    c = *p++;                   /* pick up a character */
	    if (prevchar == '\r') {     /* if prev char was cr */
		if (c == '\n')          /* if have cr,lf then just */
		   fseek(file, -1, 1);  /* smash lf on top of the cr */
		else
		   if (c == '\0')       /* if have cr,nul then */
			goto skipit;    /* just skip over the putc */
		/* else just fall through and allow it */
	    }
	    putc(c, file);
skipit:
	    prevchar = c;
	}
	return count;
}


/* When an error has occurred, it is possible that the two sides
 * are out of synch.  Ie: that what I think is the other side's
 * response to packet N is really their response to packet N-1.
 *
 * So, to try to prevent that, we flush all the input queued up
 * for us on the network connection on our host.
 *
 * We return the number of packets we flushed (mostly for reporting
 * when trace is active).
 */

/*int	f;socket to flush */
int
synchnet(int f)
{
	int i, j = 0;
	char rbuf[kdp_crashdump_pkt_size];
	struct sockaddr_in from;
	socklen_t fromlen;

	while (1) {
		(void) ioctl(f, FIONREAD, &i);
		if (i) {
			j++;
			fromlen = sizeof from;
			(void) recvfrom(f, rbuf, sizeof (rbuf), 0,
				(struct sockaddr *)&from, &fromlen);
		} else {
			return(j);
		}
	}
}
