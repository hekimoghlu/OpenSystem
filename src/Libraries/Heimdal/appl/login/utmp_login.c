/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include "login_locl.h"

RCSID("$Id$");

/* try to put something useful from hostname into dst, dst_sz:
 * full name, first component or address */

void
shrink_hostname (const char *hostname,
		 char *dst, size_t dst_sz)
{
    char local_hostname[MaxHostNameLen];
    char *ld, *hd;
    int ret;
    struct addrinfo *ai;

    if (strlen(hostname) < dst_sz) {
	strlcpy (dst, hostname, dst_sz);
	return;
    }
    gethostname (local_hostname, sizeof(local_hostname));
    hd = strchr (hostname, '.');
    ld = strchr (local_hostname, '.');
    if (hd != NULL && ld != NULL && strcmp(hd, ld) == 0
	&& hd - hostname < dst_sz) {
	strlcpy (dst, hostname, dst_sz);
	dst[hd - hostname] = '\0';
	return;
    }

    ret = getaddrinfo (hostname, NULL, NULL, &ai);
    if (ret) {
	strncpy (dst, hostname, dst_sz);
	return;
    }
    ret = getnameinfo (ai->ai_addr, ai->ai_addrlen,
		       dst, dst_sz,
		       NULL, 0,
		       NI_NUMERICHOST);
    freeaddrinfo (ai);
    if (ret) {
	strncpy (dst, hostname, dst_sz);
	return;
    }
}
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
 */#if !defined(HAVE_UTMPX_H) || (defined(WTMP_FILE) && !defined(WTMPX_FILE))

void
prepare_utmp (struct utmp *utmp, char *tty,
	      const char *username, const char *hostname)
{
    char *ttyx = clean_ttyname (tty);

    memset(utmp, 0, sizeof(*utmp));
    utmp->ut_time = time(NULL);
    strncpy(utmp->ut_line, ttyx, sizeof(utmp->ut_line));
    strncpy(utmp->ut_name, username, sizeof(utmp->ut_name));

# ifdef HAVE_STRUCT_UTMP_UT_USER
    strncpy(utmp->ut_user, username, sizeof(utmp->ut_user));
# endif

# ifdef HAVE_STRUCT_UTMP_UT_ADDR
    if (hostname[0]) {
        struct hostent *he;
	if ((he = gethostbyname(hostname)))
	    memcpy(&utmp->ut_addr, he->h_addr_list[0],
		   sizeof(utmp->ut_addr));
    }
# endif

# ifdef HAVE_STRUCT_UTMP_UT_HOST
    shrink_hostname (hostname, utmp->ut_host, sizeof(utmp->ut_host));
# endif

# ifdef HAVE_STRUCT_UTMP_UT_TYPE
    utmp->ut_type = USER_PROCESS;
# endif

# ifdef HAVE_STRUCT_UTMP_UT_PID
    utmp->ut_pid = getpid();
# endif

# ifdef HAVE_STRUCT_UTMP_UT_ID
    strncpy(utmp->ut_id, make_id(ttyx), sizeof(utmp->ut_id));
# endif
}
#endif

#ifdef HAVE_UTMPX_H
void utmp_login(char *tty, const char *username, const char *hostname)
{
    return;
}
#else

void utmp_login(char *tty, const char *username, const char *hostname)
{
    struct utmp utmp;
    int fd;

    prepare_utmp (&utmp, tty, username, hostname);

#ifdef HAVE_SETUTENT
    utmpname(_PATH_UTMP);
    setutent();
    pututline(&utmp);
    endutent();
#else

#ifdef HAVE_TTYSLOT
    {
      int ttyno;
      ttyno = ttyslot();
      if (ttyno > 0 && (fd = open(_PATH_UTMP, O_WRONLY, 0)) >= 0) {
	lseek(fd, (long)(ttyno * sizeof(struct utmp)), SEEK_SET);
	write(fd, &utmp, sizeof(struct utmp));
	close(fd);
      }
    }
#endif /* HAVE_TTYSLOT */
#endif /* HAVE_SETUTENT */

    if ((fd = open(_PATH_WTMP, O_WRONLY|O_APPEND, 0)) >= 0) {
	write(fd, &utmp, sizeof(struct utmp));
	close(fd);
    }
}

#endif /* !HAVE_UTMPX_H */

