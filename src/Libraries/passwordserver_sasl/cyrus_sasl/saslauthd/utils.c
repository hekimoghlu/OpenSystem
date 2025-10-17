/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

#include "utils.h"
#include "globals.h"


/****************************************
 * flags     global from saslauthd-main.c
 *****************************************/

/*************************************************************
 * Variadic logging function to simplify printing of log
 * messages
 **************************************************************/
void logger(int priority, const char *function, const char *format, ...) {
        va_list 	arg_list;
        char    	buffer[1024];
	static int	have_syslog = 0;

        va_start(arg_list, format);
        vsnprintf(buffer, 1023, format, arg_list);
	va_end(arg_list);

        buffer[1023] = '\0';

        if (flags & LOG_USE_STDERR)
                fprintf(stderr, L_STDERR_FORMAT, getpid(), function, buffer);

        if (flags & LOG_USE_SYSLOG) {
		if (!have_syslog) {
			openlog("saslauthd", LOG_PID|LOG_NDELAY, LOG_AUTH);
			have_syslog = 1;
		}

                syslog(priority, "%-16s: %s", function, buffer);
	}
}


/**************************************************************
 * I/O wrapper to attempt to ensure a full record gets
 * transmitted. If the function returns anything less than
 * bytesrequested, it should be considered a failure.
 **************************************************************/
ssize_t tx_rec(int filefd, void *prebuff, size_t bytesrequested) {
	int		rc;
        ssize_t		bytesio = 0;
        size_t		bytesleft = 0;
        void		*buff;

        bytesleft = bytesrequested;
        buff = prebuff;

        while (bytesleft > 0) {
                bytesio = write(filefd, buff, bytesleft);
		rc = errno;

                if (bytesio < 0) {
			logger(L_ERR, L_FUNC, "write failure");
			logger(L_ERR, L_FUNC, "write: %s", strerror(rc));
                        return(bytesio);
		}

                if (bytesio == 0 && errno != EINTR)
                        return(bytesrequested - bytesleft);

                bytesleft -= bytesio;
                buff = (void *)((char *)buff + bytesio);

        }

        return(bytesrequested);
}


/**************************************************************
 * I/O wrapper to attempt to read in the specified amount of
 * data, without any guarantees. If the function returns 
 * anything less than bytesrequested, it should be considered
 * a failure.
 **************************************************************/
ssize_t rx_rec(int filefd, void *prebuff, size_t bytesrequested) {
	int		rc;
        ssize_t		bytesio = 0;
        size_t		bytesleft = 0;
        void		*buff;

        bytesleft = bytesrequested;
        buff = prebuff;

        while (bytesleft > 0) {
                bytesio = read(filefd, buff, bytesleft);
		rc = errno;

                if (bytesio < 0) {
			logger(L_ERR, L_FUNC, "read failure");
			logger(L_ERR, L_FUNC, "read: %s", strerror(rc));
                        return(bytesio);
		}

                if (bytesio == 0 && errno != EINTR)
                        return(bytesrequested - bytesleft);

                bytesleft -= bytesio;
                buff = (void *)((char *)buff + bytesio);

        }

        return(bytesrequested);
}


/**************************************************************
 * I/O wrapper to attempt to write out the specified vector.
 * data, without any guarantees. If the function returns 
 * -1, the vector wasn't completely written.
 **************************************************************/
int retry_writev(int fd, struct iovec *iov, int iovcnt) {
	int n;               /* return value from writev() */
	int i;               /* loop counter */
	int written;         /* bytes written so far */
	static int iov_max;  /* max number of iovec entries */

#ifdef MAXIOV
	iov_max = MAXIOV;
#else 
# ifdef IOV_MAX
	iov_max = IOV_MAX;
# else 
	iov_max = 8192;
# endif 
#endif

	written = 0;

	for (;;) {

		while (iovcnt && iov[0].iov_len == 0) {
			iov++;
			iovcnt--;
		}

		if (!iovcnt) {
			return written;
		}

		n = (int)writev(fd, iov, iovcnt > iov_max ? iov_max : iovcnt);

		if (n == -1) {
			if (errno == EINVAL && iov_max > 10) {
				iov_max /= 2;
				continue;
			}

			if (errno == EINTR) {
				continue;
			}

			return -1;

		} else {
			written += n;
		}

		for (i = 0; i < iovcnt; i++) {
			if ((int) iov[i].iov_len > n) {
				iov[i].iov_base = (char *)iov[i].iov_base + n;
				iov[i].iov_len -= n;
				break;
			}

			n -= iov[i].iov_len;
			iov[i].iov_len = 0;
		}

		if (i == iovcnt) {
			return written;
		}
	}
}

#ifndef HAVE_ASPRINTF

# include <stdarg.h>

/*
 * asprintf -- work around lame systems that haven't added their own yet
 *
 * XXX relies on a valid working (SuSv3) vsnprintf(), OK on SunOS-5.10 BUT NOT BEFORE!
 */
int
asprintf(char **str,
	 const char *fmt,
	 ...)
{
	va_list ap;
	char *newstr;
	size_t len;
	int ret;

	*str = NULL;

	va_start(ap, fmt);
	ret = vsnprintf((char *) NULL, (size_t) 0, fmt, ap);
	va_end(ap);
	if (ret < 0) {
		return ret;
	}
	len = (size_t) ret + 1;		/* allow for nul */
	if ((newstr = malloc(len)) == NULL) {
		return (-1);
	}
	va_start(ap, fmt);
	ret = vsnprintf(newstr, len, fmt, ap);
	va_end(ap);
	if (ret >= 0 && (size_t) ret < len) { /* XXX (ret == len-1) */
		*str = newstr;
	} else {
		free(newstr);
		return ret;
	}

	return ret;
}
#endif /* HAVE_ASPRINTF */

#ifndef HAVE_STRLCPY
/* strlcpy -- copy string smartly.
 *
 * i believe/hope this is compatible with the BSD strlcpy(). 
 */
size_t saslauthd_strlcpy(char *dst, const char *src, size_t len)
{
	size_t n;

	if (len <= 0) {
		/* we can't do anything ! */
		return strlen(src);
	}

	/* assert(len >= 1); */
	for (n = 0; n < len-1; n++) {
		if ((dst[n] = src[n]) == '\0') break;
	}
	if (n >= len-1) {
		/* ran out of space */
		dst[n] = '\0';
		while(src[n]) n++;
	}
	return n;
}
#endif

#ifndef HAVE_STRLCAT
size_t saslauthd_strlcat(char *dst, const char *src, size_t len)
{
	size_t i, j, o;

	o = strlen(dst);
	if (len < o + 1)
		return o + strlen(src);
	len -= o + 1;
	for (i = 0, j = o; i < len; i++, j++) {
		if ((dst[j] = src[i]) == '\0') break;
	}
	dst[j] = '\0';
	if (src[i] == '\0') {
		return j;
	} else {
		return j + strlen(src + i);
	}
}
#endif
