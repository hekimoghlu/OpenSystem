/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#include <config.h>

#ifndef HAVE_FLOCK

#include "roken.h"

#define OP_MASK (LOCK_SH | LOCK_EX | LOCK_UN)


ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rk_flock(int fd, int operation)
{
#if defined(HAVE_FCNTL) && defined(F_SETLK)
  struct flock arg;
  int code, cmd;

  arg.l_whence = SEEK_SET;
  arg.l_start = 0;
  arg.l_len = 0;		/* means to EOF */

  if (operation & LOCK_NB)
    cmd = F_SETLK;
  else
    cmd = F_SETLKW;		/* Blocking */

  switch (operation & OP_MASK) {
  case LOCK_UN:
    arg.l_type = F_UNLCK;
    code = fcntl(fd, F_SETLK, &arg);
    break;
  case LOCK_SH:
    arg.l_type = F_RDLCK;
    code = fcntl(fd, cmd, &arg);
    break;
  case LOCK_EX:
    arg.l_type = F_WRLCK;
    code = fcntl(fd, cmd, &arg);
    break;
  default:
    errno = EINVAL;
    code = -1;
    break;
  }
  return code;

#elif defined(_WIN32)
  /* Windows */

#define FLOCK_OFFSET_LOW  0
#define FLOCK_OFFSET_HIGH 0
#define FLOCK_LENGTH_LOW  0x00000000
#define FLOCK_LENGTH_HIGH 0x80000000

  HANDLE hFile;
  OVERLAPPED ov;
  BOOL rv = FALSE;
  DWORD f = 0;

  hFile = (HANDLE) _get_osfhandle(fd);
  if (hFile == NULL || hFile == INVALID_HANDLE_VALUE) {
      _set_errno(EBADF);
      return -1;
  }

  ZeroMemory(&ov, sizeof(ov));
  ov.hEvent = NULL;
  ov.Offset = FLOCK_OFFSET_LOW;
  ov.OffsetHigh = FLOCK_OFFSET_HIGH;

  if (operation & LOCK_NB)
      f = LOCKFILE_FAIL_IMMEDIATELY;

  switch (operation & OP_MASK) {
  case LOCK_UN:			/* Unlock */
      rv = UnlockFileEx(hFile, 0,
			FLOCK_LENGTH_LOW, FLOCK_LENGTH_HIGH, &ov);
      break;

  case LOCK_SH:			/* Shared lock */
      rv = LockFileEx(hFile, f, 0,
		      FLOCK_LENGTH_LOW, FLOCK_LENGTH_HIGH, &ov);
      break;

  case LOCK_EX:			/* Exclusive lock */
      rv = LockFileEx(hFile, f|LOCKFILE_EXCLUSIVE_LOCK, 0,
		      FLOCK_LENGTH_LOW, FLOCK_LENGTH_HIGH,
		      &ov);
      break;

  default:
      _set_errno(EINVAL);
      return -1;
  }

  if (!rv) {
      switch (GetLastError()) {
      case ERROR_SHARING_VIOLATION:
      case ERROR_LOCK_VIOLATION:
      case ERROR_IO_PENDING:
	  _set_errno(EWOULDBLOCK);
	  break;

      case ERROR_ACCESS_DENIED:
	  _set_errno(EACCES);
	  break;

      default:
	  _set_errno(ENOLCK);
      }
      return -1;
  }

  return 0;

#else
  return -1;
#endif
}

#endif

