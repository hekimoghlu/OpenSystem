/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
/* The root directory (used when setting local-send and local-receive
   values).  */
#define ZROOTDIR "/"

/* The current directory (used by uuconv as a prefix for the newly
   created file names).  */
#define ZCURDIR "."

/* The names of the Taylor UUCP configuration files.  These are
   appended to NEWCONFIGLIB which is defined in Makefile.  */
#define CONFIGFILE "/config"
#define SYSFILE "/sys"
#define PORTFILE "/port"
#define DIALFILE "/dial"
#define CALLFILE "/call"
#define PASSWDFILE "/passwd"
#define DIALCODEFILE "/dialcode"

/* The names of the various V2 configuration files.  These are
   appended to OLDCONFIGLIB which is defined in Makefile.  */
#define V2_SYSTEMS "/L.sys"
#define V2_DEVICES "/L-devices"
#define V2_USERFILE "/USERFILE"
#define V2_CMDS "/L.cmds"
#define V2_DIALCODES "/L-dialcodes"

/* The names of the HDB configuration files.  These are appended to
   OLDCONFIGLIB which is defined in Makefile.  */
#define HDB_SYSFILES "/Sysfiles"
#define HDB_SYSTEMS "/Systems"
#define HDB_PERMISSIONS "/Permissions"
#define HDB_DEVICES "/Devices"
#define HDB_DIALERS "/Dialers"
#define HDB_DIALCODES "/Dialcodes"
#define HDB_MAXUUXQTS "/Maxuuxqts"
#define HDB_REMOTE_UNKNOWN "/remote.unknown"

/* A string which is inserted between the value of OLDCONFIGLIB
   (defined in the Makefile) and any names specified in the HDB
   Sysfiles file.  */
#define HDB_SEPARATOR "/"

/* A macro to check whether fopen failed because the file did not
   exist.  */
#define FNO_SUCH_FILE() (errno == ENOENT)

#if ! HAVE_STRERROR

/* We need a definition for strerror; normally the function in the
   unix directory is used, but we want to be independent of that
   library.  This macro evaluates its argument multiple times.  */
extern int sys_nerr;
extern char *sys_errlist[];

#define strerror(ierr) \
  ((ierr) >= 0 && (ierr) < sys_nerr ? sys_errlist[ierr] : "unknown error")

#endif /* ! HAVE_STRERROR */

/* This macro is used to make a filename found in a configuration file
   into an absolute path.  The zdir argument is the directory to put it
   in.  The zset argument is set to the new string.  The fallocated
   argument is set to TRUE if the new string was allocated.  */
#define MAKE_ABSOLUTE(zset, fallocated, zfile, zdir, pblock) \
  do \
    { \
      if (*(zfile) == '/') \
	{ \
	  (zset) = (zfile); \
	  (fallocated) = FALSE; \
	} \
      else \
	{ \
	  size_t abs_cdir, abs_cfile; \
	  char *abs_zret; \
\
	  abs_cdir = strlen (zdir); \
	  abs_cfile = strlen (zfile); \
	  abs_zret = (char *) uuconf_malloc ((pblock), \
					     abs_cdir + abs_cfile + 2); \
	  (zset) = abs_zret; \
	  (fallocated) = TRUE; \
	  if (abs_zret != NULL) \
	    { \
	      memcpy ((pointer) abs_zret, (pointer) (zdir), abs_cdir); \
	      abs_zret[abs_cdir] = '/'; \
	      memcpy ((pointer) (abs_zret + abs_cdir + 1), \
		      (pointer) (zfile), abs_cfile + 1); \
	    } \
	} \
    } \
  while (0)

/* We want to be able to mark the Taylor UUCP system files as close on
   exec.  */
#if HAVE_FCNTL_H
#include <fcntl.h>
#else
#if HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#endif

#ifndef FD_CLOEXEC
#define FD_CLOEXEC 1
#endif

#define CLOSE_ON_EXEC(e) \
  do \
    { \
      int cle_i = fileno (e); \
 \
      fcntl (cle_i, F_SETFD, fcntl (cle_i, F_GETFD, 0) | FD_CLOEXEC); \
    } \
  while (0)
