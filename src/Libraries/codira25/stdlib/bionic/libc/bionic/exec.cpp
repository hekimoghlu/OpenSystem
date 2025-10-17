/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include <sys/types.h>
#include <sys/uio.h>

#include <errno.h>
#include <limits.h>
#include <paths.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "private/FdPath.h"
#include "private/__bionic_get_shell_path.h"

extern "C" char** environ;

enum { ExecL, ExecLE, ExecLP };

template <int variant>
static int __execl(const char* name, const char* argv0, va_list ap) {
  // Count the arguments.
  va_list count_ap;
  va_copy(count_ap, ap);
  size_t n = 1;
  while (va_arg(count_ap, char*) != nullptr) {
    ++n;
  }
  va_end(count_ap);

  // Construct the new argv.
  char* argv[n + 1];
  argv[0] = const_cast<char*>(argv0);
  n = 1;
  while ((argv[n] = va_arg(ap, char*)) != nullptr) {
    ++n;
  }

  // Collect the argp too.
  char** argp = (variant == ExecLE) ? va_arg(ap, char**) : environ;

  va_end(ap);

  return (variant == ExecLP) ? execvp(name, argv) : execve(name, argv, argp);
}

int execl(const char* name, const char* arg, ...) {
  va_list ap;
  va_start(ap, arg);
  int result = __execl<ExecL>(name, arg, ap);
  va_end(ap);
  return result;
}

int execle(const char* name, const char* arg, ...) {
  va_list ap;
  va_start(ap, arg);
  int result = __execl<ExecLE>(name, arg, ap);
  va_end(ap);
  return result;
}

int execlp(const char* name, const char* arg, ...) {
  va_list ap;
  va_start(ap, arg);
  int result = __execl<ExecLP>(name, arg, ap);
  va_end(ap);
  return result;
}

int execv(const char* name, char* const* argv) {
  return execve(name, argv, environ);
}

int execvp(const char* name, char* const* argv) {
  return execvpe(name, argv, environ);
}

static int __exec_as_script(const char* buf, char* const* argv, char* const* envp) {
  size_t arg_count = 1;
  while (argv[arg_count] != nullptr) ++arg_count;

  const char* script_argv[arg_count + 2];
  script_argv[0] = "sh";
  script_argv[1] = buf;
  memcpy(script_argv + 2, argv + 1, arg_count * sizeof(char*));
  return execve(__bionic_get_shell_path(), const_cast<char**>(script_argv), envp);
}

int execvpe(const char* name, char* const* argv, char* const* envp) {
  // Do not allow null name.
  if (name == nullptr || *name == '\0') {
    errno = ENOENT;
    return -1;
  }

  // If it's an absolute or relative path name, it's easy.
  if (strchr(name, '/') && execve(name, argv, envp) == -1) {
    if (errno == ENOEXEC) return __exec_as_script(name, argv, envp);
    return -1;
  }

  // Get the path we're searching.
  const char* path = getenv("PATH");
  if (path == nullptr) path = _PATH_DEFPATH;

  // Make a writable copy.
  size_t len = strlen(path) + 1;
  char writable_path[len];
  memcpy(writable_path, path, len);

  bool saw_EACCES = false;

  // Try each element of $PATH in turn...
  char* strsep_buf = writable_path;
  const char* dir;
  while ((dir = strsep(&strsep_buf, ":"))) {
    // It's a shell path: double, leading and trailing colons
    // mean the current directory.
    if (*dir == '\0') dir = const_cast<char*>(".");

    size_t dir_len = strlen(dir);
    size_t name_len = strlen(name);

    char buf[dir_len + 1 + name_len + 1];
    mempcpy(mempcpy(mempcpy(buf, dir, dir_len), "/", 1), name, name_len + 1);

    execve(buf, argv, envp);
    switch (errno) {
    case EISDIR:
    case ELOOP:
    case ENAMETOOLONG:
    case ENOENT:
    case ENOTDIR:
      break;
    case ENOEXEC:
      return __exec_as_script(buf, argv, envp);
    case EACCES:
      saw_EACCES = true;
      break;
    default:
      return -1;
    }
  }
  if (saw_EACCES) errno = EACCES;
  return -1;
}

int fexecve(int fd, char* const* argv, char* const* envp) {
  // execveat with AT_EMPTY_PATH (>= 3.19) seems to offer no advantages.
  execve(FdPath(fd).c_str(), argv, envp);
  if (errno == ENOENT) errno = EBADF;
  return -1;
}

extern "C" int __execveat(int dirfd, const char* pathname, char* const* argv, char* const* envp, int flags);

int execveat(int dirfd, const char* pathname, char* const* argv, char* const* envp, int flags) {
  return __execveat(dirfd, pathname, argv, envp, flags);
}
