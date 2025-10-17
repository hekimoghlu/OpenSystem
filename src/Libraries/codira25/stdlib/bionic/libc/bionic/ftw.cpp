/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#include <errno.h>
#include <fts.h>
#include <ftw.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

extern "C" FTS* __fts_open(char* const*, int, int (*)(const FTSENT**, const FTSENT**));

static int do_nftw(const char* path,
                   int (*ftw_fn)(const char*, const struct stat*, int),
                   int (*nftw_fn)(const char*, const struct stat*, int, FTW*),
                   int nfds,
                   int nftw_flags) {
  // TODO: nfds is currently unused.
  if (nfds < 1) {
    errno = EINVAL;
    return -1;
  }

  // Translate to fts_open options.
  int fts_options = FTS_LOGICAL | FTS_COMFOLLOW | FTS_NOCHDIR;
  if (nftw_fn) {
    fts_options = FTS_COMFOLLOW | ((nftw_flags & FTW_PHYS) ? FTS_PHYSICAL : FTS_LOGICAL);
    if (!(nftw_flags & FTW_CHDIR)) fts_options |= FTS_NOCHDIR;
    if (nftw_flags & FTW_MOUNT) fts_options |= FTS_XDEV;
  }
  bool postorder = (nftw_flags & FTW_DEPTH) != 0;

  // Call fts_open.
  char* const paths[2] = { const_cast<char*>(path), nullptr };
  FTS* fts = __fts_open(paths, fts_options | FTS_FOR_FTW, nullptr);
  if (fts == nullptr) {
    return -1;
  }

  // Translate fts_read results into ftw/nftw callbacks.
  int error = 0;
  FTSENT* cur;
  while (error == 0 && (cur = fts_read(fts)) != nullptr) {
    int fn_flag;
    switch (cur->fts_info) {
      case FTS_D:
        // In the postorder case, we'll translate FTS_DP to FTW_DP later.
        // In the can't-access case, we'll translate FTS_DNR to FTW_DNR later.
        if (postorder || access(cur->fts_path, R_OK) == -1) continue;
        fn_flag = FTW_D;
        break;
      case FTS_DC:
        // POSIX says nftw "shall not report" directories causing loops (http://b/31152735).
        continue;
      case FTS_DNR:
        fn_flag = FTW_DNR;
        break;
      case FTS_DP:
        if (!postorder) continue;
        fn_flag = FTW_DP;
        break;
      case FTS_F:
      case FTS_DEFAULT:
        fn_flag = FTW_F;
        break;
      case FTS_NS:
      case FTS_NSOK:
        fn_flag = FTW_NS;
        break;
      case FTS_SL:
        fn_flag = FTW_SL;
        break;
      case FTS_SLNONE:
        fn_flag = (nftw_fn != nullptr) ? FTW_SLN : FTW_NS;
        break;
      default:
        error = -1;
        continue;
    }

    // Call the appropriate function.
    if (nftw_fn != nullptr) {
      FTW ftw;
      ftw.base = cur->fts_pathlen - cur->fts_namelen;
      ftw.level = cur->fts_level;
      error = nftw_fn(cur->fts_path, cur->fts_statp, fn_flag, &ftw);
    } else {
      error = ftw_fn(cur->fts_path, cur->fts_statp, fn_flag);
    }
  }

  int saved_errno = errno;
  if (fts_close(fts) != 0 && error == 0) {
    error = -1;
  } else {
    errno = saved_errno;
  }
  return error;
}

int ftw(const char* path, int (*ftw_fn)(const char*, const struct stat*, int), int nfds) {
  return do_nftw(path, ftw_fn, nullptr, nfds, 0);
}

int nftw(const char* path, int (*nftw_fn)(const char*, const struct stat*, int, FTW*),
         int nfds, int nftw_flags) {
  return do_nftw(path, nullptr, nftw_fn, nfds, nftw_flags);
}
